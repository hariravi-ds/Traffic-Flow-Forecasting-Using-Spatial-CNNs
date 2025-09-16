import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(matrix, title="Heatmap"):
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, cmap='viridis')
    plt.title(title)
    plt.show()


def compare_heatmaps(pred, actual):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(pred, cmap="viridis")
    plt.title("Predicted")
    plt.subplot(1, 2, 2)
    sns.heatmap(actual, cmap="viridis")
    plt.title("Actual")
    plt.show()
