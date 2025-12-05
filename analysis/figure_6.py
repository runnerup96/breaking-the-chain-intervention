import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import matplotlib.patches as mpatches

# %config InlineBackend.figure_format = 'retina'

# Настройка шрифтов для LaTeX-совместимости
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.titlesize": 18,
    "figure.titlesize": 18,
    "text.usetex": False,
    "font.family": "serif",
})

models = [
    "Llama-3.2-1B",
    "Qwen3-1.7B",
    "gemma-2-2b-it",
    "Llama-3.2-3B",
    "Falcon3-3B",
    "Qwen3-4B",
    "Falcon3-7B",
    "Qwen3-8B",
    "Llama-3.1-8B"
]

model_name2simple_model_name = {
    "alpindale/Llama-3.2-1B-Instruct": "llama32-1B",
    "Qwen/Qwen3-1.7B": "qwen3-1.7B",
    "google/gemma-2-2b-it": "gemma2-2B",
    "alpindale/Llama-3.2-3B-Instruct": "llama32-3B",
    "tiiuae/Falcon3-3B-Instruct": "falcon3-3B",
    "Qwen/Qwen3-4B": "qwen3-4B",
    "tiiuae/Falcon3-7B-Instruct": "falcon3-7B",
    "Qwen/Qwen3-8B": "qwen3-8B",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "llama31-8B",
}

datasets = ["tabfact_fix", "ricechem", "averitec_fix", "entailment"]
dataset_names = ['TabFact', 'RiceChem', 'AVeriTeC', 'EntailmentBank']
submetrics = ["HSVT", "Local Edits", "Global"]  # порядок подграфиков сверху вниз

def plot_hsvt_all_models_combined(
    metric_name="faithfulness",
    submetrics=submetrics,
    model_name2simple_model_name=model_name2simple_model_name,
    path='intervention_predictions',
    output_dir='frontdoor_llm_causality/plots'
):
    colors = sns.color_palette("Set2", len(datasets))

    n_subplots = len(submetrics)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 3.5 * n_subplots), sharex=True)

    if n_subplots == 1:
        axes = [axes]

    bar_width = 0.6
    group_width = len(datasets) * (bar_width + 0.2)
    group_gap = 1.0
    n_models = len(models)
    x = np.arange(n_models) * (group_width + group_gap)

    # Загрузим все данные один раз
    all_values = {}
    for submetric in submetrics:
        values_gold = []
        values_pred = []
        for model_name in model_name2simple_model_name:
            gold_vals = []
            pred_vals = []
            for ds in datasets:
                filename = f'{path}/{ds}/{model_name2simple_model_name[model_name]}.json'
                with open(filename, "r") as fp:
                    metrics = json.load(fp)["metrics"]
                gold_vals.append(metrics[metric_name]["with_gold_structure"][submetric]["mean"])
                pred_vals.append(metrics[metric_name]["with_predicted_structure"][submetric]["mean"])
            values_gold.append(gold_vals)
            values_pred.append(pred_vals)
        all_values[submetric] = (values_gold, values_pred)

    # Рисуем каждый подграфик
    for idx, submetric in enumerate(submetrics):
        ax = axes[idx]
        values_gold, values_pred = all_values[submetric]

        for i, model in enumerate(models):
            for j, ds_name in enumerate(dataset_names):
                xpos = x[i] + j * (bar_width + 0.2)

                # Gold (заливка)
                ax.bar(xpos, values_gold[i][j], bar_width,
                       color=colors[j], edgecolor="black",
                       label=f"{ds_name} (gold)" if (idx == 0 and i == 0) else "")

                # Predicted (штриховка)
                ax.bar(xpos, values_pred[i][j], bar_width,
                       facecolor="none", edgecolor="black", hatch="///",
                       label=f"{ds_name} (pred)" if (idx == 0 and i == 0) else "")

        ax.set_ylabel(submetric, fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)  # если метрики в [0,1]; можно убрать или адаптировать

    # Общая ось X — подписи моделей
    axes[-1].set_xticks(x + (len(datasets) - 1) * (bar_width + 0.2) / 2)
    axes[-1].set_xticklabels(models, rotation=30, ha="right", fontsize=18)

    gold_patch = mpatches.Patch(facecolor="white", edgecolor="black", label="Gold")
    pred_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Predicted")

    legend_struct = fig.legend([gold_patch, pred_patch],
                               ["Gold", "Predicted"],
                               loc='lower center',
                               bbox_to_anchor=(0.2, 0.04),
                               ncol=2,
                               fontsize=16,
                               frameon=True)

    # 2) Легенда для датасетов
    dataset_patches = [
        mpatches.Patch(facecolor=colors[i], edgecolor="black", label=ds_name)
        for i, ds_name in enumerate(dataset_names)
    ]

    legend_datasets = fig.legend(dataset_patches,
                                 dataset_names,
                                 loc='lower center',
                                 bbox_to_anchor=(0.65, 0.04),
                                 ncol=4,
                                 fontsize=16,
                                 frameon=True)

    # Добавляем обе легенды
    fig.add_artist(legend_struct)
    fig.add_artist(legend_datasets)

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # оставляем место под двумя легендами

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/{metric_name}_all_submetrics_combined.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


plot_hsvt_all_models_combined()