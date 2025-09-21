## Breaking the Chain ⛓️‍💥: A Causal Analysis of Faithful Reasoning in LLMs


## Code Structure Overview

This repository implements a front-door causal analysis framework for studying how interventions on intermediate reasoning steps (mediators) affect LLM predictions. The code is organized into three main components:

### 1. `make_intervention.py` - Main Orchestration Script
The central script that coordinates the entire intervention pipeline:
- **Purpose**: Runs causal interventions on LLM reasoning chains
- **Functionality**: 
  - Loads datasets and LLM models
  - Generates initial predictions with reasoning steps
  - Applies interventions to specific reasoning components
  - Generates new predictions under interventions
  - Saves results for analysis
- **Supported Datasets**: Amazon reviews, RiceChem
- **Usage**: Command-line interface with configurable model, dataset, and batch parameters

### 2. `llm_model.py` - LLM Interface
A unified interface for different language models:
- **Purpose**: Abstracts model-specific generation logic
- **Features**:
  - Automatic model family detection (currently supports Qwen3)
  - Batch text generation with configurable parameters
  - Chat template handling for conversational models
  - Device management and memory optimization
- **Supported Models**: Qwen3-4B (extensible to other families)

### 3. `datasets_for_intervention/` - Dataset and Intervention Logic
Contains dataset-specific implementations for different domains:
- **`ricechem_dataset.py`**: Chemistry question dataset loader  
- **`ricechem_intervention.py`**: Intervention logic for chemistry reasoning chains
- **Other datasets**: Averitec, TabFact, EntailmentBank (framework-ready)

Each dataset implementation provides:
- **Dataset Loading**: JSON/CSV parsing and preprocessing
- **Prompt Construction**: Structured reasoning templates
- **Intervention Logic**: Methods to modify specific reasoning steps
- **Validation**: Ensuring intervention quality and consistency

## How It Works
1. **Initial Generation**: LLM generates predictions with explicit reasoning steps
2. **Intervention**: Specific reasoning components are systematically modified
3. **Re-generation**: New predictions are generated under interventions
4. **Analysis**: Causal effects of reasoning changes on final predictions are measured

This framework enables researchers to study how different reasoning patterns influence LLM decision-making through controlled interventions.

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
- **`evaluation_dataset`**: Choose from `"ricechem"` or `"amazon_reviews"`
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
intervention_analysis/intervention_predictions/{dataset_name}/{model_name}_{timestamp}.json
```

The output contains:
- Original model predictions with reasoning steps
- Intervention results for each reasoning component
- Final predictions under each intervention
- Validation status and any failed interventions


