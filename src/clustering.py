import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def run_kmeans(X, n_clusters=3):
    flat = X.reshape(X.shape[0], -1)
    scaled = StandardScaler().fit_transform(flat)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    return kmeans, clusters, scaled


def plot_cluster_centers(kmeans, n_clusters=3, seq_len=6, grid_size=20):
    centers = kmeans.cluster_centers_.reshape(
        n_clusters, seq_len, grid_size, grid_size)
    for i in range(n_clusters):
        plt.figure(figsize=(15, 3))
        for t in range(seq_len):
            plt.subplot(1, seq_len, t+1)
            sns.heatmap(centers[i, t], cmap='viridis', cbar=False)
            plt.title(f'Cluster {i} - T{t+1}')
        plt.show()


def plot_pca(scaled, clusters):
    reduced = PCA(n_components=2).fit_transform(scaled)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='viridis')
    plt.title("Traffic Heatmap Clusters (PCA)")
    plt.show()
