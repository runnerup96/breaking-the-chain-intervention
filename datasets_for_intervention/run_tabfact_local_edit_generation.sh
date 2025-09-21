#!/bin/bash

# --- Конфигурация (ЗАМЕНИТЕ ЗНАЧЕНИЯ НА СВОИ!) ---

# Параметры API
OPENROUTER_API_KEY="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"  # <-- ЗАМЕНИТЕ НА ВАШ РЕАЛЬНЫЙ КЛЮЧ OpenRouter
MODEL="openai/gpt-4o"
API_LINK="https://openrouter.ai/api/v1"

# Пути к данным
QUERIES_JSON="/home/chaichuk/frontdoor_llm_causality/statics/result_splits/Table-Fact-Checking/bootstrap/bootstrap_new.json"
TABLES_DIR="/home/chaichuk/frontdoor_llm_causality/statics/result_splits/Table-Fact-Checking/data/all_csv"

# Параметры выполнения
OUTPUT_PATH="/home/chaichuk/frontdoor_llm_causality/statics/result_splits/Table-Fact-Checking/bootstrap/local_edits_verified_by_table.json"  # Имя выходного файла
NUM_THREADS=10                                     # Количество параллельных запросов (5-10 оптимально)
MAX_SAMPLES=-1                                # Установите число (например, 10) для теста, или "None" для полного запуска

# --- Запуск скрипта ---
echo "Starting Local Edits generation for TabFact dataset..."
echo "Output will be saved to: $OUTPUT_PATH"

python datasets_for_intervention/generate_tabfact_local_edits.py \
    --model "$MODEL" \
    --api_link "$API_LINK" \
    --token "$OPENROUTER_API_KEY" \
    --queries_json "$QUERIES_JSON" \
    --tables_dir "$TABLES_DIR" \
    --output_path "$OUTPUT_PATH" \
    --num_threads "$NUM_THREADS" \
    --max_samples $MAX_SAMPLES  # Обратите внимание: без кавычек для None

# Проверка на ошибки
if [ $? -eq 0 ]; then
    echo "✅ Generation completed successfully!"
else
    echo "❌ Generation failed. Please check the logs above."
fi

echo "Script finished."