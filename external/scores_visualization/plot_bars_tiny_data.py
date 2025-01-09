import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def read_data(file_path):
    return pd.read_csv(file_path)

def plot_bars_tiny_data(df):
    fig, ax = plt.subplots(figsize=(16, 8))
    model_colors = {"retinanet": "orange", "fcos": "blue", "FasterRCNN": "green"}
    freeze_hatch = {True: "//",  False: ""}
    bar_width = 0.2
    x_labels = sorted(df["dataset_size"].unique())
    x_indices = np.arange(len(x_labels))

    for j, (model, model_subset) in enumerate(df.groupby("model")):
        for i, (frozen, subset) in enumerate(model_subset.groupby("freeze_backbone")):
            means = [subset[subset["dataset_size"] == size]["test_mAP"].mean() for size in x_labels]
            ax.bar(
                x_indices + j * bar_width + i * (bar_width / 2),
                means,
                bar_width / 2,
                label=f"{model} (backbone {'frozen' if frozen else 'trainable'})",
                color=model_colors.get(model, "gray"),
                hatch=freeze_hatch.get(frozen, ""),
                edgecolor="black"
            )

    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Test mAP")
    ax.set_title("Test mAP by dataset size, model, and backbone freezing status")
    ax.set_xticks(x_indices + (len(model_colors) * bar_width) / 4)
    ax.set_xticklabels(x_labels)
    ax.legend(title="Model & backbone freeze", loc="upper left")

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot test mAP by dataset size, model, and backbone freezing status.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing tiny data.')
    args = parser.parse_args()
    
    df = read_data(args.file_path)
    plot_bars_tiny_data(df)

if __name__ == '__main__':
    main()