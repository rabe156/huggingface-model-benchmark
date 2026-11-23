import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def compare_models(metrics_by_model):
    rows=[]
    for model,metrics in metrics_by_model.items(): 
        row = {
                "model": model,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"]
            }
        rows.append(row)
    comparaison=pd.DataFrame(rows)
    return comparaison


def compare_performance(benchmark_by_model):
    dataframes=[]
    for model ,dataframe in benchmark_by_model.items():
        temp=dataframe.copy()
        temp.insert(0,"model",model)
        dataframes.append(temp)
    comparison=pd.concat(dataframes , ignore_index=True)
    return comparison

def export_results(metrics_df,performance_df,distilbert_errors,bert_errors,output_dir="./results"):
    output_path=Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True)
    metrics_df.to_csv(output_path /"model_comparison.csv" , index=False)
    performance_df.to_csv(output_path /"performance_comparison.csv" , index=False)
    distilbert_errors.to_csv(output_path /"distilbert_errors.csv" , index=False)
    bert_errors.to_csv(output_path /"bert_errors.csv" , index=False)



def plot_model_metrics(metrics_df, output_dir="./results/figures"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract data
    models = metrics_df["model"]
    accuracy = metrics_df["accuracy"]
    f1 = metrics_df["f1"]

    # Bar positions
    x = np.arange(len(models))
    width = 0.35

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5))

    # Grouped bars
    ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy"
    )

    ax.bar(
        x + width / 2,
        f1,
        width,
        label="F1-score"
    )

    # Configure chart
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Classification Performance")
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    # Save chart
    fig.savefig(
        output_path / "model_metrics.png",
        dpi=300
    )

    plt.close(fig)


def plot_throughput(performance_df, output_dir="results/figures"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot one line per model
    for model_name, group in performance_df.groupby("model"):

        group = group.sort_values("batch_size")

        ax.plot(
            group["batch_size"],
            group["throughput"],
            marker="o",
            label=model_name
        )

    # Configure chart
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (samples/second)")
    ax.set_title("Inference Throughput by Batch Size")

    ax.set_xticks(sorted(performance_df["batch_size"].unique()))
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    # Save chart
    fig.savefig(
        output_path / "throughput_comparison.png",
        dpi=300
    )

    plt.close(fig)

