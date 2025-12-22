import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle, Patch
from pathlib import Path
from typing import Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import resample

# sns.set_theme(style="whitegrid", font_scale=1.5)
# %config InlineBackend.figure_format = 'retina'

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


"""
Example metric values:

"metrics": {
        "performance": {
            "with_gold_structure": {
                "score_match": {
                    "mean": 0.961764705882353,
                    "std": 0.19176380367885928
                }
            },
            "with_predicted_structure": {
                "proof_match": {
                    "mean": 0.0029411764705882353,
                    "std": 0.05415280188094697
                },
                "score_match": {
                    "mean": 0.9647058823529412,
                    "std": 0.18452220166303673
                }
            }
        },
        "faithfullness": {
            "with_gold_structure": {
                "HSVT": {
                    "mean": 0.9705882352941176,
                    "std": 0.1689577248981773
                },
                "Local Edits": {
                    "mean": 0.010784313725490196,
                    "std": 0.10328607022711436
                },
                "Global": {
                    "mean": 0.011764705882352941,
                    "std": 0.10782531046954917
                }
            },
            "with_predicted_structure": {
                "HSVT": {
                    "mean": 0.95,
                    "std": 0.21794494717703367
                },
                "Local Edits": {
                    "mean": 0.05,
                    "std": 0.21794494717703367
                },
                "Global": {
                    "mean": 0.04411764705882353,
                    "std": 0.20535647123189618
                }
            }
        },
        "local_edit_influence": {
            "with_gold_structure": {
                "delete": {
                    "mean": 0.011764705882352941,
                    "std": 0.10782531046954917
                },
                "replace": {
                    "mean": 0.0029411764705882353,
                    "std": 0.05415280188094697
                },
                "rewire": {
                    "mean": 0.01764705882352941,
                    "std": 0.13166487815058467
                }
            },
            "with_predicted_structure": {
                "delete": {
                    "mean": 0.04411764705882353,
                    "std": 0.20535647123189618
                },
                "replace": {
                    "mean": 0.047058823529411764,
                    "std": 0.21176470588235294
                },
                "rewire": {
                    "mean": 0.058823529411764705,
                    "std": 0.23529411764705882
                }
            }
        }
    },
"""

SIMPLE_MODEL_NAME2_PLOT_NAME = {
    "llama31-8B": "Llama 3.1 8B",
    "llama32-3B": "Llama 3.2 3B",
    "llama32-1B": "Llama 3.2 1B",
    "qwen3-1.7B": "Qwen 3 1.7B",
    "qwen3-4B": "Qwen 3 4B",
    "qwen3-8B": "Qwen 3 8B",
    "gemma2-2B": "Gemma 2 2B",
    "gemma2-9B": "Gemma 2 9B",
    "falcon3-3B": "Falcon 3 3B",
    "falcon3-7B": "Falcon 3 7B",
}

FAMILY2MARKER = {
    "llama": "o",
    "falcon": "P",
    "qwen": "D",
    "gemma": "p"
}

ORDERED_DATASETS = ["tabfact", "ricechem", "averitec", "entailment"]
DATASET_NAMES = ['TabFact', 'RiceChem', 'AVeriTeC', 'EntailmentBank']

def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

def collect_metrics(root_dir: str):
    """
    Collect the metrics for the intervention predictions.
    """
    dataset_directories = list(d for d in Path(root_dir).glob("*") if d.is_dir())

    dataset2model2metrics = {}

    for dataset_dir in dataset_directories:
        dataset2model2metrics[dataset_dir.name] = {}
        for model_results_filename in dataset_dir.glob("*_metrics_only.json"):
            model_name = (model_results_filename.stem).split("_")[0]
            results = load_json(model_results_filename)
            dataset2model2metrics[dataset_dir.name][model_name] = results
    return dataset2model2metrics

def get_model_family(model_name: str):
    """
    Get the family of the model.
    """
    if "llama" in model_name:
        return "llama"
    elif "falcon" in model_name:
        return "falcon"
    elif "qwen" in model_name:
        return "qwen"
    elif "gemma" in model_name:
        return "gemma"

def get_model_size(model_name: str):
    """
    Get the size of the model.
    """
    size = model_name.split("-")[-1]
    size = size.split("B")[0]
    return float(size)

def nested_metrics_to_df(nested_metrics: dict):
    """
    Convert the nested metrics to a dataframe.
    """
    performance_records = []
    faithfullness_records = []
    local_edit_influence_records = []

    for dataset in nested_metrics:
        for model in nested_metrics[dataset]:
            for metric_type in nested_metrics[dataset][model]:
                assert metric_type in {"performance", "faithfullness", "faithfulness", "local_edit_influence"}, "Unexpected metric type: " + str(metric_type)

                for structure in nested_metrics[dataset][model][metric_type]:
                    assert structure in {"with_gold_structure", "with_predicted_structure"}, "Unexpected structure: " + str(structure)
                    for intervention_or_metric_name in nested_metrics[dataset][model][metric_type][structure]:
                        if metric_type == "local_edit_influence":
                            continue

                        if intervention_or_metric_name == "correct_predictions_count":
                            continue

                        intervention_or_metric_name_to_plot = intervention_or_metric_name
                        if intervention_or_metric_name in ("score_match", "verdict_match", "label_match"):
                            intervention_or_metric_name_to_plot = "score_match"

                        if intervention_or_metric_name in ("checklist_match", "proof_match", "expression_match", "structure_match"):
                            intervention_or_metric_name_to_plot = "structure_match"

                        mean = nested_metrics[dataset][model][metric_type][structure][intervention_or_metric_name]["mean"]
                        std = nested_metrics[dataset][model][metric_type][structure][intervention_or_metric_name]["std"]

                        model_name_to_plot = SIMPLE_MODEL_NAME2_PLOT_NAME[model]

                        print(dataset, model, metric_type, structure, intervention_or_metric_name)
                        if metric_type == "performance":
                            performance_records.append({
                                "dataset": dataset,
                                "model": model_name_to_plot,
                                "model_family": get_model_family(model),
                                "model_size": get_model_size(model),
                                "structure": structure,
                                "metric": intervention_or_metric_name_to_plot,
                                "mean": mean,
                                "std": std
                            })
                        elif metric_type == "faithfulness" or metric_type == "faithfullness":
                            faithfullness_records.append({
                                "dataset": dataset,
                                "model": model_name_to_plot,
                                "model_family": get_model_family(model),
                                "model_size": get_model_size(model),
                                "structure": structure,
                                "intervention": intervention_or_metric_name_to_plot,
                                "mean": mean,
                                "std": std
                            })
                        elif metric_type == "local_edit_influence":
                            pass # TODO: add local edit influence records

    return pd.DataFrame(performance_records), pd.DataFrame(faithfullness_records), pd.DataFrame(local_edit_influence_records)

 # Function to calculate R² with confidence interval using bootstrap
def calculate_r2_ci(x, y, n_bootstrap=1000, ci=95):
    """Calculate R² score with confidence interval using bootstrap resampling."""
    if len(x) < 3:  # Need at least 3 points for meaningful regression
        return np.nan, np.nan, np.nan
    
    # Original R² score
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    y_pred = lr.predict(x.reshape(-1, 1))
    original_r2 = r2_score(y, y_pred)
    
    # Bootstrap R² scores
    bootstrap_r2s = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(range(len(x)), n_samples=len(x))
        x_boot = x[indices]
        y_boot = y[indices]
        
        # Fit model and calculate R²
        lr_boot = LinearRegression()
        lr_boot.fit(x_boot.reshape(-1, 1), y_boot)
        y_pred_boot = lr_boot.predict(x_boot.reshape(-1, 1))
        r2_boot = r2_score(y_boot, y_pred_boot)
        bootstrap_r2s.append(r2_boot)
    
    # Calculate confidence interval
    alpha = (100 - ci) / 2
    lower_ci = np.percentile(bootstrap_r2s, alpha)
    upper_ci = np.percentile(bootstrap_r2s, 100 - alpha)
    
    return original_r2, lower_ci, upper_ci


def average_by_datasets(df: pd.DataFrame):
    # Filter out models which have less than the maximum number of entries
    model_value_counts = df["model"].value_counts()
    max_count = max(model_value_counts.values)

    for model in model_value_counts[model_value_counts < max_count].index:
        df = df[df["model"] != model]

    # Automatically detect column types and create aggregation functions
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns
    
    # Remove groupby columns from aggregation
    groupby_cols = ["model", "structure", "metric"] if "metric" in df.columns else ["model", "structure", "intervention"]
    numeric_columns = [col for col in numeric_columns if col not in groupby_cols]
    categorical_columns = [col for col in categorical_columns if col not in groupby_cols]
    
    # Create aggregation dictionary
    agg_functions = {}
    for col in numeric_columns:
        agg_functions[col] = 'mean'
    for col in categorical_columns:
        agg_functions[col] = 'first'
    
    average_df = df.groupby(groupby_cols).agg(agg_functions).reset_index()
    average_df.drop(columns=["dataset"], inplace=True)
    return average_df

def gold_vs_predicted_structure_plot(df: pd.DataFrame, output_dir: str, title: str, only_keep_positive_change: bool = False, intervention_type: Optional[str] = None):
    """
    Plot the performance or faithfullness of the models for the gold and predicted structure.
    """
    df = df.sort_values(by=["model_size"], ascending=[True])
    
    if "metric" in df.columns:
        # Separate gold and predicted structure data
        gold_df = df[
            (df['structure'] == 'with_gold_structure')
            & (df['metric'] == 'score_match')
        ].copy().sort_values(["model_size", "model"], ascending=[True, True]).reset_index(drop=True)
        predicted_df = df[
            (df['structure'] == 'with_predicted_structure')
            & (df['metric'] == 'score_match')
        ].copy().sort_values(["model_size", "model"], ascending=[True, True]).reset_index(drop=True)
    elif "intervention" in df.columns:
        assert intervention_type is not None, "Intervention type is required with 'intervention' column"
        gold_df = df[
            (df['structure'] == 'with_gold_structure')
            & (df['intervention'] == intervention_type)
        ].copy().sort_values(["model_size", "model"], ascending=[True, True]).reset_index(drop=True)
        predicted_df = df[
            (df['structure'] == 'with_predicted_structure')
            & (df['intervention'] == intervention_type)
        ].copy().sort_values(["model_size", "model"], ascending=[True, True]).reset_index(drop=True)
    else:
        raise ValueError("Expected 'metric' or 'intervention' columns in dataframe")

    change = predicted_df['mean'] - gold_df['mean']
    if only_keep_positive_change:
        mask = change > 0.03
        gold_df = gold_df.loc[mask].reset_index(drop=True)
        predicted_df = predicted_df.loc[mask].reset_index(drop=True)
        change = change.loc[mask].reset_index(drop=True)
    
    # Create the plot
    models = gold_df['model'].unique()
    if only_keep_positive_change:
        figsize = (6, 5)
    else:
        figsize = (12, 5) if len(models) <= 7 else (14, 5)
    fig, ax = plt.subplots(figsize=figsize)
    
    family2set1colors = dict(zip(gold_df["model_family"].unique(), sns.color_palette("Set1", len(models))))
    # Get unique models and their positions
    models = gold_df['model'].unique()
    assert set(models) == set(predicted_df['model'].unique()), "Models in gold and predicted dataframes are not the same"
    x_positions = np.arange(len(models))

    # Plot overlapping bars with transparency
    gold_bars = ax.bar(x_positions, gold_df['mean'], edgecolor="black", alpha=0.6, label='Gold', 
                       color=gold_df["model_family"].map(family2set1colors), width=0.6)
    predicted_bars = ax.bar(x_positions, predicted_df['mean'], edgecolor="black", facecolor="none", hatch="///",
                            alpha=0.6, label='Predicted', width=0.6)
    
    # Add arrows showing the change from gold to predicted
    for i, model in enumerate(models):
        gold_val = gold_df[gold_df['model'] == model]['mean'].iloc[0] if len(gold_df[gold_df['model'] == model]) > 0 else 0
        pred_val = predicted_df[predicted_df['model'] == model]['mean'].iloc[0] if len(predicted_df[predicted_df['model'] == model]) > 0 else 0
        
        # Calculate percentage change
        change = pred_val - gold_val
        percent_change = (change / gold_val * 100) if gold_val != 0 else 0
        arrow_color = 'tab:gray' if abs(percent_change) < 1 else 'tab:green' if percent_change > 0 else 'tab:red'
        
        # Add arrow from gold to predicted value
        if abs(percent_change) > 5:  # Only show arrow if there's a meaningful change
            ax.annotate('', xy=(i, pred_val), xytext=(i, gold_val),
                       arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5, alpha=0.8))
        else:
            ax.annotate('', xy=(i, pred_val), xytext=(i, gold_val),
                       arrowprops=dict(arrowstyle='-', color=arrow_color, lw=2.5, alpha=0.8))    
        # Add percentage change label
        label_y = max(gold_val, pred_val) * 1.03
        ax.text(i, label_y, f'{percent_change:+.1f}%', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=arrow_color, alpha=0.9)
    
    # Customize the plot
    ax.set_ylabel('Performance')
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=20, ha="right")
    # ax.legend(bbox_to_anchor=(0.5, -0.20), loc='upper center', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.get_ylim()
    
    y_min, y_max = ax.get_ylim()
    # Adjust for the percentage change label
    ax.set_ylim(y_min, y_max * 1.03)

    # Legend 1: Structure type
    structure_legend_elements = [
        Patch(facecolor='white', edgecolor='black', alpha=0.6, label='Gold'),
        Patch(facecolor='none', edgecolor='black', hatch='///', alpha=0.6, label='Predicted')
    ]

    # Legend 2: Model family colors
    model_family_legend_elements = [
        Patch(facecolor=color, edgecolor='black', alpha=0.6, label=family.capitalize())
        for family, color in family2set1colors.items()
    ]

    # Position legends below the plot
    structure_legend = ax.legend(handles=structure_legend_elements, 
                                bbox_to_anchor=(0.1, -0.20), loc='upper center', 
                                ncol=2, title="Structure Type")

    model_family_legend = ax.legend(handles=model_family_legend_elements,
                                bbox_to_anchor=(0.85, -0.20), loc='upper center',
                                ncol=4, title="Model Family")

    ax.add_artist(structure_legend)
    ax.add_artist(model_family_legend)

    # Save the plot
    os.makedirs(f"{output_dir}/performance", exist_ok=True)
    title_filename = title.lower().replace(" ", "_").replace("\n", "_").replace(":", "_")
    plt.savefig(f"{output_dir}/performance/{title_filename}.png", dpi=300, bbox_inches="tight", bbox_extra_artists=(structure_legend, model_family_legend))
    plt.close()

def create_r2_latex_table(r2_results: dict, output_dir: str, title: str):
    """
    Create a LaTeX table with R² values and confidence intervals per dataset.
    
    Args:
        r2_results: Dictionary with dataset names as keys and (r2, r2_lower, r2_upper) tuples as values
        output_dir: Output directory for the LaTeX file
        title: Title for the table
    """
    # LaTeX table header
    latex_content = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{title}}}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Dataset & $R^2$ & 95\\% CI \\\\",
        "\\midrule"
    ]
    
    # Add rows for each dataset
    for dataset in ORDERED_DATASETS:
        if dataset in r2_results:
            r2, r2_lower, r2_upper = r2_results[dataset]
            dataset_display_name = DATASET_NAMES[ORDERED_DATASETS.index(dataset)]
            
            if not np.isnan(r2):
                latex_content.append(
                    f"{dataset_display_name} & {r2:.3f} & [{r2_lower:.3f}, {r2_upper:.3f}] \\\\"
                )
            else:
                latex_content.append(
                    f"{dataset_display_name} & N/A & [insufficient data] \\\\"
                )
    
    # Add overall R² if available
    if "overall" in r2_results:
        r2, r2_lower, r2_upper = r2_results["overall"]
        latex_content.extend([
            "\\midrule",
            f"Overall & {r2:.3f} & [{r2_lower:.3f}, {r2_upper:.3f}] \\\\"
        ])
    
    # LaTeX table footer
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    # Write to file
    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    title_filename = title.lower().replace(" ", "_").replace("\n", "_").replace(":", "_").replace("(", "").replace(")", "")
    latex_file_path = f"{output_dir}/tables/{title_filename}_r2_table.tex"
    
    with open(latex_file_path, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX table saved to: {latex_file_path}")


def final_answer_vs_structure_performance_plot(performance_df: pd.DataFrame, output_dir: str, title: str):
    """
    Plot the performance of the models for the final answer and structure.
    """
    performance_df = performance_df
    
    # Separate final answer and structure data
    final_answer_df = performance_df[
        (performance_df['structure'] == 'with_predicted_structure')
        & (performance_df['metric'] == 'score_match')
    ].copy().sort_values("model_size", ascending=[True]).reset_index(drop=True)
    structure_df = performance_df[
        (performance_df['structure'] == 'with_predicted_structure')
        & (performance_df['metric'] == 'structure_match')
    ].copy().sort_values("model_size", ascending=[True]).reset_index(drop=True)
    merged_data = final_answer_df.merge(
        structure_df[["model", "dataset", "mean"]].rename(columns={"mean": "structure_mean_score"}), 
        on=["model", "dataset"], 
        how="inner"
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    # Get unique datasets and create color mapping
    unique_datasets = merged_data['dataset'].unique()
    assert set(unique_datasets) == set(ORDERED_DATASETS), "Datasets in merged data are not the same as ordered datasets"
    dataset_colors = dict(zip(ORDERED_DATASETS, sns.color_palette("Set2", len(unique_datasets))))
    
    for family in final_answer_df['model_family'].unique():
        family_data = merged_data[merged_data['model_family'] == family]
        for dataset in family_data['dataset'].unique():
            dataset_family_data = family_data[family_data['dataset'] == dataset]
            sns.scatterplot(
                y=dataset_family_data['mean'], x=dataset_family_data['structure_mean_score'], 
                color=dataset_colors[dataset], ax=ax, s=100, marker=FAMILY2MARKER[family]
            )
    # Add regression lines with confidence intervals for each dataset and collect R² values
    r2_results = {}
    
    for dataset in unique_datasets:
        dataset_data = merged_data[merged_data['dataset'] == dataset]
        
        # Add regression line
        sns.regplot(
            data=dataset_data, 
            x='structure_mean_score', 
            y='mean',
            color=dataset_colors[dataset],
            scatter=False,  # Don't plot points again
            ax=ax,
            line_kws={'linewidth': 1},
            ci=95  # 95% confidence interval
        )
        
        # Calculate and store R² values
        x = dataset_data['structure_mean_score'].values
        y = dataset_data['mean'].values
        r2, r2_lower, r2_upper = calculate_r2_ci(x, y)
        r2_results[dataset] = (r2, r2_lower, r2_upper)
    
    ax.set_ylim(0, 1.05)
    
    # Create dataset legend (color-based)
    dataset_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=DATASET_NAMES[ORDERED_DATASETS.index(dataset)])
                      for dataset, color in dataset_colors.items()]
    dataset_legend = ax.legend(handles=dataset_handles, title="Dataset", ncols=2,
                              bbox_to_anchor=(-0.1, -0.2), loc='upper left')
    
    # Create model family legend (marker-based)
    family_handles = [plt.Line2D([0], [0], marker=marker, color='gray', markersize=12, label=family.capitalize(), linestyle='None')
                     for family, marker in FAMILY2MARKER.items()]
    family_legend = ax.legend(handles=family_handles, title="Model Family", ncols=2,
                            bbox_to_anchor=(0.55, -0.2), loc='upper left')
    
    # Add both legends to the plot
    ax.add_artist(dataset_legend)
    ax.add_artist(family_legend)
    
    ax.set_xlabel("Structure prediction accuracy")
    ax.set_ylabel("Final answer accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Create LaTeX table with R² results
    create_r2_latex_table(r2_results, output_dir, title)
    
    # Print R² results to console for quick reference
    print(f"\n=== R² Scores for {title} ===")
    print("Dataset\t\tR²\t\t95% CI")
    print("-" * 50)
    
    for dataset in ORDERED_DATASETS:
        if dataset in r2_results:
            r2, r2_lower, r2_upper = r2_results[dataset]
            dataset_display_name = DATASET_NAMES[ORDERED_DATASETS.index(dataset)]
            
            if not np.isnan(r2):
                print(f"{dataset_display_name:<15}\t{r2:.3f}\t\t[{r2_lower:.3f}, {r2_upper:.3f}]")
            else:
                print(f"{dataset_display_name:<15}\tN/A\t\t[insufficient data]")
    
    title_filename = title.lower().replace(" ", "_").replace("\n", "_").replace(":", "_")
    os.makedirs(f"{output_dir}/performance", exist_ok=True)
    plt.savefig(f"{output_dir}/performance/{title_filename}.png", dpi=300, 
                bbox_inches="tight", bbox_extra_artists=(dataset_legend, family_legend))
    plt.close()

def performance_faithfulness_pareto_front_plot(performance_df: pd.DataFrame, faithfulness_df: pd.DataFrame, output_dir: str, title: str, structure_type: str):
    """
    Plot the performance and faithfulness of the models for the gold and predicted structure.
    """
    performance_df = performance_df[
        (performance_df["structure"] == structure_type)
        & (performance_df["metric"] == "score_match")
    ].copy().sort_values("model_size", ascending=[True]).reset_index(drop=True)
    
    faithfulness_df = faithfulness_df[
        faithfulness_df["structure"] == structure_type
    ].copy()
    # Filter out Qwen3-4B and gemma2-9B since they are not available for some datasets
    faithfulness_df = faithfulness_df[
        (faithfulness_df["model"] != "qwen3-4B") & (faithfulness_df["model"] != "gemma2-9B")
    ].copy()

    hsvt_df = faithfulness_df[faithfulness_df["intervention"] == "HSVT"]
    local_edits_df = faithfulness_df[faithfulness_df["intervention"] == "Local Edits"]
    global_df = faithfulness_df[faithfulness_df["intervention"] == "Global"]
    assert len(hsvt_df) == len(local_edits_df) == len(global_df) == len(performance_df)

    fig, ax = plt.subplots(figsize=(9, 6))

    intervention_colors = {"HSVT": "tab:blue", "Local Edits": "tab:orange", "Global": "tab:green"}
    
    # Merge dataframes to ensure proper alignment
    merged_data = performance_df.merge(
        hsvt_df[["model", "mean"]].rename(columns={"mean": "hsvt_mean"}), 
        on="model", how="inner"
    ).merge(
        local_edits_df[["model", "mean"]].rename(columns={"mean": "local_edits_mean"}), 
        on="model", how="inner"
    ).merge(
        global_df[["model", "mean"]].rename(columns={"mean": "global_mean"}), 
        on="model", how="inner"
    )
    
    for family in merged_data["model_family"].unique():
        family_data = merged_data[merged_data["model_family"] == family]
        
        sns.scatterplot(
            y=family_data["mean"], x=family_data["hsvt_mean"], 
            color="tab:blue", ax=ax, s=100, marker=FAMILY2MARKER[family]
        )
        sns.scatterplot(
            y=family_data["mean"], x=family_data["local_edits_mean"], 
            color="tab:orange", ax=ax, s=100, marker=FAMILY2MARKER[family]
        )
        sns.scatterplot(
            y=family_data["mean"], x=family_data["global_mean"], 
            color="tab:green", ax=ax, s=100, marker=FAMILY2MARKER[family]
        )
    
    # Create intervention legend (color-based)
    intervention_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=intervention)
                           for intervention, color in intervention_colors.items()]
    intervention_legend = ax.legend(handles=intervention_handles, title="Intervention Type", 
                                  bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Create model family legend (marker-based)
    family_handles = [plt.Line2D([0], [0], marker=marker, color='gray', markersize=12, label=family.capitalize(), linestyle='None')
                     for family, marker in FAMILY2MARKER.items()]
    family_legend = ax.legend(handles=family_handles, title="Model Family", 
                            bbox_to_anchor=(1.05, 0.6), loc='upper left')
    
    # Add both legends to the plot
    ax.add_artist(intervention_legend)
    ax.add_artist(family_legend)
            
    plt.ylim(0, 1.05)
    ax.set_xlabel("Faithfulness")
    ax.set_ylabel("Performance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    title_filename = title.lower().replace(" ", "_").replace("\n", "_").replace(":", "_")
    os.makedirs(f"{output_dir}/performance", exist_ok=True)
    plt.savefig(f"{output_dir}/performance/{title_filename}.png", dpi=300,
                bbox_inches="tight", bbox_extra_artists=(intervention_legend, family_legend))
    plt.close()

if __name__ == "__main__":
    results_dir = "intervention_predictions"
    output_dir = "breaking-the-chain-intervention/plots"

    dataset2model2metrics = collect_metrics(results_dir)
    performance_df, faithfullness_df, local_edit_influence_df = nested_metrics_to_df(dataset2model2metrics)
    
    final_answer_vs_structure_performance_plot(performance_df, output_dir, "Final answer vs structure prediction accuracy")

    average_performance_df = average_by_datasets(performance_df)
    gold_vs_predicted_structure_plot(average_performance_df, output_dir, "Performance with gold and predicted structure\n(averaged by datasets)")

    dataset_name = "entailment"
    # for dataset_name in performance_df["dataset"].unique():
    dataset_performance_df = performance_df[performance_df["dataset"] == dataset_name]
    dataset_name = DATASET_NAMES[ORDERED_DATASETS.index(dataset_name)]
    gold_vs_predicted_structure_plot(
        dataset_performance_df,
        output_dir,
        f"{dataset_name}: gold vs predicted structure\n(only positive changes)",
        only_keep_positive_change=True
    )

    # average_performance_df = average_by_datasets(performance_df)
    average_faithfullness_df = average_by_datasets(faithfullness_df)

    performance_faithfulness_pareto_front_plot(average_performance_df, average_faithfullness_df, output_dir, "Performance vs faithfulness\n(with gold structure, averaged by datasets)", "with_gold_structure")
    performance_faithfulness_pareto_front_plot(average_performance_df, average_faithfullness_df, output_dir, "Performance vs faithfulness\n(with predicted structure, averaged by datasets)", "with_predicted_structure")

    gold_vs_predicted_structure_plot(average_faithfullness_df, output_dir, "HSVT faithfulness, gold vs predicted structure\n(averaged by datasets)", intervention_type="HSVT")
    gold_vs_predicted_structure_plot(average_faithfullness_df, output_dir, "Local Edits faithfulness, gold vs predicted structure\n(averaged by datasets)", intervention_type="Local Edits")
    gold_vs_predicted_structure_plot(average_faithfullness_df, output_dir, "Global faithfulness, gold vs predicted structure\n(averaged by datasets)", intervention_type="Global")