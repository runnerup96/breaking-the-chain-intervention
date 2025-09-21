import os
import json
import argparse
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
import threading
import queue
import openai
import httpx
from httpx_socks import SyncProxyTransport
from typing import Callable, List, Dict, Any
from openrouter_batch_api import OpenrouterBatchApiClass
from tabfact_dataset import TabFactDataset

LOCAL_EDIT_PROMPT_TEMPLATE = """
You are an expert in logical reasoning and table data analysis.
Your task is to generate semantically altered versions of a given Domain Specific Language (DSL) expression for the TabFact dataset. 
The altered expressions must be GUARANTEED TO EVALUATE TO FALSE, while preserving the overall structure and making only minimal, localized changes.

### INSTRUCTIONS
1.  **Input**: You will be given a table (in CSV format with '#' as separator), a claim (statement), and the original DSL expression that correctly verifies the claim (evaluates to TRUE).
2.  **Output**: Generate 5 distinct variants of the original expression. Each variant should:
    *   Make 1 to 3 minimal, localized changes (e.g., swap a column name with a semantically similar one, swap a value with another valid value from the table, change a comparison operator, or combine 2-3 such changes).
    *   Be syntactically valid according to the DSL.
    *   Be GUARANTEED to evaluate to FALSE for the given table.
    *   Preserve the overall structure and function nesting of the original expression as much as possible.
3.  **DSL Functions** (Use these for reference):
    - `greater{A, B}`: A is greater than B, return True, other return False
    - `hop{Row, Field Name}`: Hop to the Field name column in the Row.
    - `count{C}`: Counting how many rows are in the given C Rows.
    - `eq{A, B}`: A is equal to B, return True, other return False
    - `and{A, B, ...}`: Logical AND operation, return True if all arguments are True, otherwise return False
    - `only{C}`: Check if the given set of rows C contains exactly one row, return True if so, otherwise return False
    - `diff{A, B}`: Calculate the difference between A and B (A - B)
    - `avg{C}`: Calculate the average value of the specified field across the given set of rows C
    - `all_greater{C, Value}`: Check if all values in the specified field across the given set of rows C are greater than the given Value, return True if so
    - `sum{C}`: Calculate the sum of the values in the specified field across the given set of rows C
    - `all_eq{C, Value}`: Check if all values in the specified field across the given set of rows C are equal to the given Value, return True if so
    - `filter_eq{C, Field Name, Value}`: Filter the set of rows C to include only those where the specified Field Name equals the given Value
    - `filter_greater{C, Field Name, Value}`: Filter the set of rows C to include only those where the specified Field Name is greater than the given Value
    - `filter_not_eq{C, Field Name, Value}`: Filter the set of rows C to include only those where the specified Field Name is not equal to the given Value
    - `filter_less{C, Field Name, Value}`: Filter the set of rows C to include only those where the specified Field Name is less than the given Value
    - `argmax{C, Field Name}`: Return the row from the set C that has the maximum value in the specified Field Name
    - `argmin{C, Field Name}`: Return the row from the set C that has the minimum value in the specified Field Name
    - `max{C}`: Find the maximum value in the specified field across the given set of rows C
    - `min{C}`: Find the minimum value in the specified field across the given set of rows C
    - `filter_greater_eq{C, Field Name, Value}`: Filter the set of rows C to include only those where the specified Field Name is greater than or equal to the given Value
    - `filter_less_eq{C, Field Name, Value}`: Filter the set of rows C to include only those where the specified Field Name is less than or equal to the given Value
    - `all_greater_eq{C, Value}`: Check if all values in the specified field across the given set of rows C are greater than or equal to the given Value, return True if so
    - `all_less{C, Value}`: Check if all values in the specified field across the given set of rows C are less than the given Value, return True if so
    - `not_eq{A, B}`: A is not equal to B, return True, other return False

### OUTPUT FORMAT
Return ONLY a JSON object with a key "edits". The value is a list of 5 objects. Each object has two keys:
- "expression": The altered DSL expression (string), ending with `=True`.
- "explanation": A short string (1-2 sentences) explaining WHAT was changed and WHY it GUARANTEES that THE RESULT IS FALSE. Mention that the change is type-safe.

**CRITICAL**: Changes must be TYPE-SAFE. Never compare a string to a number or use an invalid field. All arguments to functions must remain logically and semantically valid within the context of the table.

### FEW-SHOT EXAMPLES

Example 1:
Table (CSV with '#' separator):
rank#athlete#nation#gold
1#Usain Bolt#Jamaica#2
2#Shawn Crawford#United States#1
Claim: Usain Bolt won more gold medals than Shawn Crawford.
Original Expression: greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True
Output: 
{
  "edits": [
    {
      "expression": "greater{hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}; hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}}=True",
      "explanation": "Swapped 'Usain Bolt' and 'Shawn Crawford'. Crawford has 1 gold, Bolt has 2, so Crawford > Bolt is false. Type-safe: comparing numbers."
    },
    {
      "expression": "less{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True",
      "explanation": "Changed 'greater' to 'less'. Bolt has 2 gold, Crawford has 1, so 2 < 1 is false. Type-safe: comparing numbers."
    },
    {
      "expression": "eq{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True",
      "explanation": "Changed 'greater' to 'eq'. Bolt has 2 gold, Crawford has 1, so 2 == 1 is false. Type-safe: comparing numbers."
    },
    {
      "expression": "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}}=True",
      "explanation": "Changed second athlete from 'Shawn Crawford' to 'Usain Bolt'. Comparing Bolt's gold (2) to himself (2) with 'greater' yields 2 > 2, which is false. Type-safe: comparing numbers."
    },
    {
      "expression": "less{hop{filter_eq{all_rows; nation; Jamaica}; gold}; hop{filter_eq{all_rows; nation; United States}; gold}}=True",
      "explanation": "Replaced 'athlete' with 'nation' and used country names. Changed operator to 'less'. Jamaica (2) < USA (1) is false. Type-safe: comparing numbers from the 'gold' column."
    }
  ]
}

Example 2:
Table (CSV with '#' separator):
player#team#goals
Messi#PSG#30
Ronaldo#AlNassr#25
Neymar#AlHilal#20
Claim: Ronaldo scored more goals than Messi.
Original Expression: greater{hop{filter_eq{all_rows; player; Ronaldo}; goals}; hop{filter_eq{all_rows; player; Messi}; goals}}=True
Output: 
{
  "edits": [
    {
      "expression": "greater{hop{filter_eq{all_rows; player; Neymar}; goals}; hop{filter_eq{all_rows; player; Messi}; goals}}=True",
      "explanation": "Changed first player from 'Ronaldo' to 'Neymar'. Neymar has 20 goals, Messi has 30, so 20 > 30 is false. Type-safe: comparing numbers."
    },
    {
      "expression": "eq{hop{filter_eq{all_rows; player; Ronaldo}; goals}; hop{filter_eq{all_rows; player; Messi}; goals}}=True",
      "explanation": "Changed 'greater' to 'eq'. Ronaldo has 25 goals, Messi has 30, so 25 == 30 is false. Type-safe: comparing numbers."
    },
    {
      "expression": "greater{hop{filter_eq{all_rows; player; Ronaldo}; goals}; hop{filter_eq{all_rows; player; Ronaldo}; goals}}=True",
      "explanation": "Changed second player from 'Messi' to 'Ronaldo'. Comparing Ronaldo's goals (25) to himself (25) with 'greater' yields 25 > 25, which is false. Type-safe: comparing numbers."
    },
    {
      "expression": "greater{hop{filter_eq{all_rows; team; AlNassr}; goals}; hop{filter_eq{all_rows; team; PSG}; goals}}=True",
      "explanation": "Replaced 'player' with 'team' and used team names. Type-safe: comparing numbers."
    },
    {
      "expression": "eq{hop{filter_eq{all_rows; player; Ronaldo}; team}; hop{filter_eq{all_rows; player; Messi}; team}}=True",
      "explanation": "Changed comparison from 'goals' to 'team' and operator from 'greater' to 'eq' (AlNassr == PSG is false). Type-safe: comparing strings."
    }
  ]
}

Example 3:
Table (CSV with '#' separator):
product#category#price#rating
iPhone#Electronics#999#4.8
Book#Stationery#15#4.5
Laptop#Electronics#1200#4.7
Claim: There is an electronics product priced over $1000 with a rating above 4.5.
Original Expression: and{greater{hop{filter_eq{all_rows; product; Laptop}; price}; 1000}; greater{hop{filter_eq{all_rows; product; Laptop}; rating}; 4.5}}=True
Output: 
{
  "edits": [
    {
      "expression": "and{greater{hop{filter_eq{all_rows; product; iPhone}; price}; 1000}; greater{hop{filter_eq{all_rows; product; Laptop}; rating}; 4.5}}=True",
      "explanation": "Changed first product from 'Laptop' to 'iPhone'. iPhone price (999) > 1000 is false. False AND True = False. Type-safe: comparing numbers."
    },
    {
      "expression": "and{greater{hop{filter_eq{all_rows; product; Laptop}; price}; 1000}; greater{hop{filter_eq{all_rows; product; Laptop}; rating}; 5.0}}=True",
      "explanation": "Changed rating threshold from 4.5 to 5.0. Laptop rating (4.7) > 5.0 is false. True AND False = False. Type-safe: comparing numbers."
    },
    {
      "expression": "and{greater{hop{filter_eq{all_rows; product; Laptop}; price}; 1500}; greater{hop{filter_eq{all_rows; product; Laptop}; rating}; 4.5}}=True",
      "explanation": "Changed price threshold from 1000 to 1500. Laptop price (1200) > 1500 is false. False AND True = False. Type-safe: comparing numbers."
    },
    {
      "expression": "and{less{hop{argmax{filter_eq{all_rows; category; Electronics}; price}; price}; 1000}; greater{hop{filter_eq{all_rows; product; Laptop}; rating}; 4.5}}=True",
      "explanation": "Changed first filter from 'product; Laptop' to 'category; Electronics'. 'filter_eq' returns multiple rows, 'hop' on multiple rows is undefined/invalid. Using 'argmax' and `less` (1200 < 1000 is false). Type-safe."
    },
    {
      "expression": "or{greater{hop{filter_eq{all_rows; product; iPhone}; price}; 1000}; greater{hop{filter_eq{all_rows; product; Book}; rating}; 4.5}}=True",
      "explanation": "Changed 'and' to 'or' and used 'iPhone' and 'Book'. iPhone price > 1000 is false, Book rating > 4.5 is false. False OR False = False. Type-safe: comparing numbers."
    }
  ]
}

Now generate 5 edits for the following input:
"""

def parse_llm_response(response_text: str) -> List[Dict[str, str]]:
    import json
    try:
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        data = json.loads(response_text)
        if "edits" in data and isinstance(data["edits"], list):
            parsed_edits = []
            for item in data["edits"]:
                if isinstance(item, dict) and "expression" in item and "explanation" in item:
                    parsed_edits.append({
                        "expression": item["expression"].strip(),
                        "explanation": item["explanation"].strip()
                    })
            return parsed_edits
        else:
            print(f"[ERROR] 'edits' key not found or not a list in JSON.")
            return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error during parsing: {e}")
        return []


def validate_expression_syntax(expr: str) -> bool:
    if not (expr.endswith("=True") or expr.endswith("=False")):
        return False
    brace_count = 0
    for char in expr:
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count < 0:
                return False
    return brace_count == 0


def main():
    parser = argparse.ArgumentParser(description="Generate Local Edits for TabFact dataset using LLM.")
    parser.add_argument("--model", type=str, default="openai/gpt-4o", help="Model name for OpenRouter API")
    parser.add_argument("--api_link", type=str, default="https://openrouter.ai/api/v1", help="API endpoint")
    parser.add_argument("--token", type=str, required=True, help="API token")
    parser.add_argument("--queries_json", type=str, required=True, help="Path to queries JSON file")
    parser.add_argument("--tables_dir", type=str, required=True, help="Path to tables directory")
    parser.add_argument("--output_path", type=str, default="tabfact_local_edits_by_table.json", help="Path to save output JSON")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of concurrent threads")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")

    args = parser.parse_args()

    # Load dataset
    print("Loading TabFact dataset...")
    dataset = TabFactDataset(queries_json_path=args.queries_json, tables_dir=args.tables_dir)
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    # Prepare samples
    samples = deepcopy(dataset.data)
    if args.max_samples != -1:
        samples = samples[:args.max_samples]
        print(f"Processing first {args.max_samples} samples.")

    # Add unique index for API client
    for i, sample in enumerate(samples):
        sample['unique_idx'] = f"{sample['idx']}_gen_{i}"

    transport = SyncProxyTransport.from_url(f'socks5://chaichuk:tG83uScS8oe8@193.124.46.176:8080')
    http_client = httpx.Client(transport=transport)

    # Initialize API client
    api_client = OpenrouterBatchApiClass(
        model=args.model,
        api_link=args.api_link,
        token=args.token,
        num_threads=args.num_threads,
        http_client=http_client,
        max_tokens=2048
    )

    # Define prompt function
    def build_prompt(sample):
      return f"""{LOCAL_EDIT_PROMPT_TEMPLATE}
Table (CSV with '#' separator):
{sample['table_html_csv']}

Claim:
{sample['statement']}

Original Expression: {sample['verifier_query_gt']}"""

    # Generate edits
    print("Starting LLM queries...")
    raw_results = api_client.call(
        samples=samples,
        prompt_func=build_prompt,
        sample_idx_key="unique_idx"
    )

    # --- Ключевое изменение: Группировка по table_id ---
    # Инициализируем словарь для хранения результатов по таблицам
    results_by_table = {}

    # Создаем вспомогательный словарь для быстрого поиска sample по его idx
    sample_idx_to_sample = {s['idx']: s for s in samples}

    # Парсим и верифицируем результаты
    for result in tqdm(raw_results, desc="Parsing and grouping by table_id"):
        # Восстанавливаем оригинальный sample_idx (без суффикса '_gen_X')
        original_sample_idx = result['sample_idx'].split('_gen_')[0]

        # Находим исходный сэмпл в датасете по его idx
        original_sample = sample_idx_to_sample.get(original_sample_idx)
        if not original_sample:
            print(f"[WARNING] Sample with idx {original_sample_idx} not found in dataset. Skipping.")
            continue

        # Получаем table_id из найденного сэмпла
        table_id = original_sample['table_id']

        # Инициализируем запись для таблицы, если её нет
        if table_id not in results_by_table:
            results_by_table[table_id] = []

        # Создаем запись для текущего сэмпла
        sample_entry = {
            "main_question": original_sample['statement'],
            "table_name": original_sample.get('table_caption', table_id),
            "main_program": original_sample['verifier_query_gt'],
            "local_edits": [],    # Сюда положим сгенерированные интервенции
            "raw_response": result['response']['response']  # Для отладки
        }

        # Парсим ответ LLM
        if "Error:" not in result['response']['response']:
            parsed_edits = parse_llm_response(result['response']['response'])
            valid_edits = [edit for edit in parsed_edits if validate_expression_syntax(edit['expression'])]
            sample_entry["local_edits"] = valid_edits
        else:
            sample_entry["local_edits"] = []
            sample_entry["error"] = "API Error"

        # Добавляем запись в список для этой таблицы
        results_by_table[table_id].append(sample_entry)

    # --- Форматируем выходные данные ---
    # Создаем список для финального JSON, как в вашем примере
    output_list = []
    for table_id, entries in results_by_table.items():
        # Создаем одну запись на таблицу
        table_entry = {
            "table_id": table_id,
            "entries": entries  # Список всех сэмплов для этой таблицы
        }
        output_list.append(table_entry)

    # Save results
    output_data = {
        "metadata": {
            "model": args.model,
            "generated_at": datetime.now().isoformat(),
            "total_tables": len(results_by_table),
            "total_samples": len(raw_results),
            "total_valid_edits": sum(len(entry["local_edits"]) for table in output_list for entry in table["entries"])
        },
        "data": output_list  # Основной массив данных
    }

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Results saved to {args.output_path}")
    print(f"Total tables processed: {output_data['metadata']['total_tables']}")
    print(f"Total valid edits generated: {output_data['metadata']['total_valid_edits']}")

if __name__ == "__main__":
    main()