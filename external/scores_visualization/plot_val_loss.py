import pandas as pd
import matplotlib.pyplot as plt
import argparse

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df.head(58)

def plot_val_loss(df):
    plt.figure(figsize=(12, 8))
    plt.plot(df['epoch'], df['def-detr-5 - val_mAP'], label='Deformable DETR', marker='o')
    plt.plot(df['epoch'], df['retinanet - val_mAP'], label='RetinaNet', marker='s')
    plt.plot(df['epoch'], df['fcos - val_mAP'], label='FCOS', marker='^')
    plt.plot(df['epoch'], df['faster r-cnn - val_mAP'], label='Faster R-CNN', marker='D')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation mAP', fontsize=12)
    plt.title('Convergence speed across various models', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0, 0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot validation mAP across models.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing validation mAP data.')
    args = parser.parse_args()
    
    df = read_data(args.file_path)
    plot_val_loss(df)

if __name__ == '__main__':
    main()