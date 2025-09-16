import numpy as np
import keras_tuner as kt
from utils import save_model
from clustering import run_kmeans, plot_cluster_centers, plot_pca
from visualization import compare_heatmaps
from model import build_model
from data_prep import load_distance_data, map_sensors, create_synthetic_data, build_sequences
import sys
sys.path.append("../src")


# load distance + map sensors
df, sensors, dist_matrix = load_distance_data("../data/distance.csv")
sensor_grid_positions = map_sensors(dist_matrix, sensors, grid_size=20)

# create synthetic traffic
traffic_df = create_synthetic_data(sensors, n_timesteps=100)
X, Y = build_sequences(traffic_df, sensor_grid_positions,
                       grid_size=20, seq_len=6)

# reshape for CNN
X_in = X[..., np.newaxis]
Y_flat = Y.reshape((Y.shape[0], -1))

# tune & train
tuner = kt.RandomSearch(
    build_model, objective='val_loss',
    max_trials=5, executions_per_trial=1,
    overwrite=True, directory='tuning_logs', project_name='traffic_cnn'
)
tuner.search(X_in, Y_flat, epochs=10, validation_split=0.2)
best_model = tuner.get_best_models(1)[0]

loss, mae = best_model.evaluate(X_in, Y_flat)
print("Test Loss:", loss, "MAE:", mae)

# visualize prediction
pred = best_model.predict(X_in[0:1])[0].reshape(20, 20)
compare_heatmaps(pred, Y[0])

# clustering
kmeans, clusters, scaled = run_kmeans(X, n_clusters=3)
plot_cluster_centers(kmeans, 3, seq_len=6, grid_size=20)
plot_pca(scaled, clusters)

save_model(best_model, "../models/cnn_model.h5")
