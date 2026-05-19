## Breaking the Chain ⛓️‍💥: A Causal Analysis of LLM Faithfulness to Intermediate Structures

## Code Structure Overview

This repository implements a front-door causal analysis framework for studying how interventions on intermediate reasoning steps (mediators) affect LLM predictions. The code is organized into three main components:

### 1. `make_intervention.py` - Main Orchestration Script

The central script that coordinates the entire intervention pipeline:

- **Purpose**: Runs causal interventions on LLM reasoning chains
- **Functionality**:
  - Loads datasets and LLM models
  - Generates initial predictions with structured reasoning
  - Applies interventions to specific reasoning components
  - Generates new predictions under interventions
  - Saves results for analysis
- **Supported Dataset with structured mediator**: [RiceChem](https://github.com/luffycodes/Automated-Long-Answer-Grading), [TabFact](https://github.com/wenhuchen/Table-Fact-Checking), [AVeriTeC](https://fever.ai/dataset/averitec.html), [CRUXEval](https://github.com/facebookresearch/cruxeval)
- **Usage**: Command-line interface with configurable model, dataset, and batch parameters

### 2. `llm_model.py` - LLM Interface

A unified interface for different language models:

- **Purpose**: Abstracts model-specific generation logic
- **Features**:
  - Automatic model family detection (currently supports Qwen3)
  - Batch text generation with configurable parameters
  - Chat template handling for conversational models
  - Device management and memory optimization
- **Supported Models**: Qwen, Gemma, Llama, Falcon with pre-trained or API functionality

### 3. `datasets_for_intervention/` - Dataset and Intervention Logic

Contains dataset-specific implementations for different domains:

- **`*_dataset.py`**: dataset loader
- **`*_intervention.py`**: Dataset-specific intervention logic
- **`*_structure_processor.py`**: Dataset-specific tool and mediator parser
- **`*_evaluation.py`**: Evaluation script to eval the model faithfulness and performance

Each dataset implementation provides:

- **Dataset Loading**: JSON/CSV parsing and preprocessing
- **Prompt Construction**: Structured reasoning templates
- **Intervention Logic**: Methods to modify specific reasoning steps
- **Validation**: Ensuring intervention quality and consistency

## Downloading the datasets (`statics/`)

This project expects the dataset artifacts to be present under:

${PROJECT_PATH}/statics/datasets/
AVeriTeC/...
RiceChem/...
TabFact/...

We host the redistributable files on the Hugging Face Hub in a companion dataset repository:

- **HF dataset repo:** `THunderCondOR/breaking-the-chain-intervention-data`

**Note (TabFact):** TabFact contains **>10,000 CSV files** in `TabFact/data/all_csv/`. Because the Hugging Face Hub enforces a **10k files per directory** limit, we store this folder as a single archive (`datasets/TabFact/data/all_csv.tar.gz`). The download helper below will automatically extract it back into `${PROJECT_PATH}/statics/datasets/TabFact/data/all_csv/` so the on-disk layout matches the original one.

### Option A (recommended): one-command download + auto-extract

1) Install the minimal dependency:

```bash
pip install -U huggingface_hub
```

2) Make sure `PROJECT_PATH` is set to the root of this repository:

```bash
export PROJECT_PATH=/path/to/breaking-the-chain-intervention
```

3) Download everything into `${PROJECT_PATH}/statics` (and auto-extract TabFact):

```bash
python download_datasets.py \
  --repo_id THunderCondOR/breaking-the-chain-intervention-data \
  --all
```

After this, you should have the same structure as the original `statics/` layout, including:

- `${PROJECT_PATH}/statics/datasets/AVeriTeC/...`
- `${PROJECT_PATH}/statics/datasets/TabFact/data/all_csv/...`

> The script downloads into a temporary folder under `statics/` and cleans it up at the end, so you should not end up with a persistent `.cache` directory in `statics/`.

### Option B: download only selected datasets

Examples:

Download **only TabFact**:

```bash
python download_datasets.py \
  --repo_id THunderCondOR/breaking-the-chain-intervention-data \
  --only tabfact
```

Download **AVeriTeC + TabFact**:

```bash
python download_datasets.py \
  --repo_id THunderCondOR/breaking-the-chain-intervention-data \
  --only averitec tabfact
```

#### Keep TabFact compressed (skip extraction)

If you want to keep `all_csv.tar.gz` without unpacking:

```bash
python download_datasets.py \
  --repo_id THunderCondOR/breaking-the-chain-intervention-data \
  --only tabfact \
  --no_extract
```

### Notes on CRUXEval

CRUXEval is loaded automatically from HuggingFace
([`cruxeval-org/cruxeval`](https://huggingface.co/datasets/cruxeval-org/cruxeval))
on first run and cached to
`${PROJECT_PATH}/statics/datasets/CRUXEval/test.jsonl`.

The mediator for CRUXEval is the **execution trace** of the function
(line-by-line locals). The Local Edit interventions correspond to the 6
deterministic perturbation levels described in the analysis notebook:

| Level | Name                  | Effect on M                                     |
| ----- | --------------------- | ----------------------------------------------- |
| 1     | SingleValueMutation   | One variable's value mutated, same type         |
| 2     | DoubleValueMutation   | Two variables mutated                           |
| 3     | TypeCoercion          | One variable coerced to a different type        |
| 4     | StepDrop              | One step removed from the trace                 |
| 5     | StepDuplicate         | One mid-trace step duplicated                   |
| 6     | StepReorder           | Two adjacent steps swapped                      |

For each Local Edit, `expected_target_after_intervention` is computed
deterministically by evaluating the function's `return` expression against
the locals of the **perturbed** last step (`simulate_from_trace`). The model
is faithful to the mediator iff its answer matches this expected value.
The evaluator reports additional per-level faithfulness (`S(k)`). 

### Notes on RiceChem redistribution

**RiceChem is not redistributed** in this repository due to licensing restrictions. The HF package may include an empty placeholder directory.
If you want to run RiceChem experiments, download it from the original source and place the files under:

```
${PROJECT_PATH}/statics/datasets/RiceChem/
```

## How It Works

1. **Initial Generation**: LLM generates predictions with explicit reasoning steps
2. **Intervention**: Specific reasoning components are systematically modified
3. **Re-generation**: New predictions are generated under interventions
4. **Analysis**: Causal effects of reasoning changes on final predictions are measured

This framework enables researchers to study how different reasoning patterns influence LLM decision-making through controlled interventions.

## Generated Figures

The `analysis/` folder contains visualization scripts and generated figures from the paper that illustrate the results of the intervention experiments and overall model performance.

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster inference)

### Installation Steps

1. **Create a virtual environment**:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n breaking-the-chain-env python=3.12
conda activate breaking-the-chain-env
```

3. **Install dependencies**:

```bash
# Install using pip
pip install -r requirements.txt

# Or using uv (faster, if available)
uv pip install -r requirements.txt
```

4. **Verify installation via testing**:

```bash
python -m pytest datasets_for_intervention/test_intervention
```

### Alternative way

You can create conda env from `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate breaking-the-chain-env
```

## How to Run

### Using the Shell Script (Recommended)

The easiest way to run interventions is using the provided shell script:

```bash
# Make the script executable
chmod +x make_intervention_script.sh

# Run the script
./make_intervention_script.sh
```

**Before running, modify the script to match your setup:**

- **`project_path`**: Set to your project directory path
- **`python_path`**: Path to project interpreter
- **`evaluation_dataset`**: Choose from `"ricechem"`, `"tabfact"`, `"averitec"`, `"cruxeval"`
- **`model_name`**: Specify the LLM model (e.g., `"Qwen/Qwen3-4B"`)
- **`batch_size`**: Adjust based on your GPU memory (default: 32)
- **`CUDA_DEVICE_NUMBER`**: Set your GPU device number

```bash
export PROJECT_PATH="/path/to/your/project"
export CUDA_VISIBLE_DEVICES=0

python make_intervention.py \
    --model_name "Qwen/Qwen3-4B" \
    --evaluation_dataset "ricechem" \
    --batch_size 32
```

Results are saved to:

```
intervention_analysis/intervention_predictions/{dataset_name}/{prompting_regime}/{model_name}_{timestamp}.json
```

The output contains:

- Original model predictions with reasoning steps
- Intervention results for each reasoning component
- Final predictions under each intervention
- Validation status and any failed interventions
