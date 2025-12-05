import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import matplotlib.patches as mpatches

# % config
# InlineBackend.figure_format = 'retina'

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.titlesize": 16,
    "figure.titlesize": 16,
    "text.usetex": False,
    "font.family": "serif",
})

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

# --- КЛЮЧЕВЫЕ ПАРАМЕТРЫ ВЫРАВНИВАНИЯ ---
Y_LABEL = "Faithfulness Score (avg)"
Y_LIM = (0, 1.0)
DPI = 300

# Позиция области графика (left, bottom, width, height) — ОДИНАКОВАЯ для обоих!
AXES_POS = [0.14, 0.22, 0.84, 0.72]  # [left, bottom, width, height] в долях от фигуры

# Легенда — фиксированное положение относительно axes
LEGEND_POS = (0.5, -0.3)  # относительно центра осей

BAR_WIDTH = 0.6
HATCH_COLOR = "black"
GRID_ALPHA = 0.3

gold_patch = mpatches.Patch(facecolor="white", edgecolor="black", label="Gold")

# штрихованный прямоугольник для Predicted
pred_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Predicted")


def plot_hsvt(metric_name="faithfulness",
              path='intervention_predictions',
              output_dir='frontdoor_llm_causality/plots'):
    colors = sns.color_palette("Set2", len(datasets))
    fig = plt.figure(figsize=(5.5, 4))
    ax = fig.add_axes(AXES_POS)

    gold_means, pred_means = [], []
    for ds in datasets:
        gold_vals, pred_vals = [], []
        for model_name in model_name2simple_model_name:
            filename = f'{path}/{ds}/{model_name2simple_model_name[model_name]}.json'
            with open(filename, "r") as fp:
                metrics = json.load(fp)["metrics"]
            gold_vals.append(metrics[metric_name]["with_gold_structure"]["HSVT"]["mean"])
            pred_vals.append(metrics[metric_name]["with_predicted_structure"]["HSVT"]["mean"])
        gold_means.append(np.mean(gold_vals))
        pred_means.append(np.mean(pred_vals))

    x = np.arange(len(datasets))
    bars_gold = ax.bar(x, gold_means, BAR_WIDTH, color=colors, alpha=0.9, edgecolor="black", label="Gold")
    bars_pred = ax.bar(x, pred_means, BAR_WIDTH, color="none", edgecolor=HATCH_COLOR,
                       hatch="//", linewidth=0.7, label="Predicted")

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=20, ha="right", fontsize=16)
    ax.set_ylabel('HSVT Faithfulness', fontsize=16)
    ax.set_ylim(Y_LIM)
    ax.grid(True, alpha=GRID_ALPHA)

    # легенды: gold/pred слева, датасеты справа
    legend1 = ax.legend([gold_patch, pred_patch], ["Gold", "Predicted"],
                        loc="upper center", bbox_to_anchor=(0.05, -0.3), ncol=1,
                        frameon=True, fontsize=12)
    ax.add_artist(legend1)

    legend2 = ax.legend(bars_gold, dataset_names,
                        loc="upper center", bbox_to_anchor=(0.62, -0.3), ncol=2,
                        frameon=True, fontsize=12)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(f"{output_dir}/{metric_name}_HSVT.{ext}", dpi=DPI,
                        bbox_inches="tight")
    plt.show()


def plot_local_global(metric_name="faithfulness",
                      path='intervention_predictions',
                      output_dir='frontdoor_llm_causality/plots'):
    submetrics = ["Local Edits", "Global"]
    base_colors = sns.color_palette("Set2", len(datasets))

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes(AXES_POS)

    gold_means, pred_means = {ds: [] for ds in datasets}, {ds: [] for ds in datasets}
    for ds in datasets:
        for sub in submetrics:
            gold_vals, pred_vals = [], []
            for model_name in model_name2simple_model_name:
                filename = f'{path}/{ds}/{model_name2simple_model_name[model_name]}.json'
                with open(filename, "r") as fp:
                    metrics = json.load(fp)["metrics"]
                gold_vals.append(metrics[metric_name]["with_gold_structure"][sub]["mean"])
                pred_vals.append(metrics[metric_name]["with_predicted_structure"][sub]["mean"])
            gold_means[ds].append(np.mean(gold_vals))
            pred_means[ds].append(np.mean(pred_vals))

    spacing = 0.4
    x = np.arange(len(datasets)) * (BAR_WIDTH * 2.2 + spacing)

    bars_gold, bars_pred = [], []
    for i, ds in enumerate(datasets):
        for j, sub in enumerate(submetrics):
            xpos = x[i] + j * (BAR_WIDTH + 0.1)
            base_color = base_colors[i]
            color = tuple(min(1, c + 0.15 * j) for c in base_color)

            bg = ax.bar(xpos, gold_means[ds][j], BAR_WIDTH,
                        color=color, alpha=0.9, edgecolor="black",
                        label=f"{sub} Gold" if i == 0 else "")
            bp = ax.bar(xpos, pred_means[ds][j], BAR_WIDTH,
                        color="none", edgecolor=HATCH_COLOR,
                        hatch="//", linewidth=0.7,
                        label=f"{sub} Pred" if i == 0 else "")
            bars_gold.append(bg[0]);
            bars_pred.append(bp[0])

    ax.set_xticks(x + BAR_WIDTH / 2 + 0.05)
    ax.set_xticklabels(dataset_names, rotation=20, ha="right", fontsize=16)
    ax.set_ylabel('Local & Global Faithfulness', fontsize=16)
    ax.set_ylim(Y_LIM)
    ax.grid(True, alpha=GRID_ALPHA)

    # Легенды в один ряд
    legend1 = ax.legend([gold_patch, pred_patch], ["Gold", "Predicted"],
                        loc="upper center", bbox_to_anchor=(0.02, -0.3), ncol=1,
                        frameon=True, fontsize=10)

    # ax.add_artist(legend1)

    # показываем все цвета для local/global
    # legend2 = ax.legend([bars_gold[i] for i in range(0, len(bars_gold), 2)] +
    #                     [bars_gold[i] for i in range(1, len(bars_gold), 2)],
    #                     ["Local"]*len(datasets) + ["Global"]*len(datasets),
    #                     loc="upper center", bbox_to_anchor=(0.62, -0.3), ncol=len(datasets),
    #                     frameon=True, edgecolor="black")

    legend_labels = []
    legend_handles = []
    for i, ds in enumerate(datasets):
        legend_handles.append(bars_gold[2 * i])  # Local bar
        legend_labels.append(f"{dataset_names[i]} Local")
        legend_handles.append(bars_gold[2 * i + 1])  # Global bar
        legend_labels.append(f"{dataset_names[i]} Global")

    legend2 = ax.legend(legend_handles, legend_labels,
                        loc="upper center", bbox_to_anchor=(0.45, -0.3),
                        ncol=3, frameon=True, fontsize=14)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(f"{output_dir}/{metric_name}_LocalGlobal.{ext}", dpi=DPI,
                        bbox_inches="tight")
    plt.show()


plot_hsvt()
plot_local_global()
