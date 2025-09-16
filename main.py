import os
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_prep import load_distance_data, map_sensors, create_synthetic_data, build_sequences
from src.model import build_model
from src.clustering import run_kmeans, plot_cluster_centers, plot_pca
from src.utils import ensure_dir


def save_heatmap(pred, actual, outpath="results/pred_vs_actual.png"):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.heatmap(pred, cmap="viridis")
    plt.title("Predicted")

    plt.subplot(1, 2, 2)
    sns.heatmap(actual, cmap="viridis")
    plt.title("Actual")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"📊 Saved heatmap comparison → {outpath}")


def main():
    ensure_dir("results")
    ensure_dir("models")

    # 1. Load distance data & map sensors
    df, sensors, dist_matrix = load_distance_data("data/distance.csv")
    sensor_grid_positions = map_sensors(dist_matrix, sensors, grid_size=20)

    # 2. Generate synthetic traffic data
    traffic_df = create_synthetic_data(sensors, n_timesteps=100)

    # 3. Build sequences
    X, Y = build_sequences(
        traffic_df, sensor_grid_positions, grid_size=20, seq_len=6)
    X_in = X[..., np.newaxis]
    Y_flat = Y.reshape((Y.shape[0], -1))

    # 4. CNN hyperparameter tuning
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=5,
        executions_per_trial=1,
        overwrite=True,
        directory="tuning_logs",
        project_name="traffic_cnn",
    )
    tuner.search(X_in, Y_flat, epochs=10, validation_split=0.2)
    best_model = tuner.get_best_models(1)[0]

    # 5. Evaluate
    loss, mae = best_model.evaluate(X_in, Y_flat)
    print(f"✅ CNN Evaluation - Loss: {loss:.4f}, MAE: {mae:.4f}")

    # 6. Visualize & save one prediction
    pred = best_model.predict(X_in[0:1])[0].reshape(20, 20)
    save_heatmap(pred, Y[0], "results/pred_vs_actual.png")

    # 7. Clustering with KMeans
    kmeans, clusters, scaled = run_kmeans(X, n_clusters=3)
    plot_cluster_centers(kmeans, 3, seq_len=6, grid_size=20)
    plt.savefig("results/cluster_centers.png", dpi=300)
    plt.close()

    plot_pca(scaled, clusters)
    plt.savefig("results/pca_clusters.png", dpi=300)
    plt.close()
    print("📊 Saved clustering results → results/")

    # 8. Save trained model (new Keras format)
    model_path = "models/cnn_model.keras"
    best_model.save(model_path)
    print(f"💾 Model saved at {model_path}")


if __name__ == "__main__":
    main()
