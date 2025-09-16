import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def load_distance_data(path):
    df = pd.read_csv(path, header=0, names=[
                     "Sensor-A", "Sensor-B", "Distance"])
    sensors = pd.unique(df[['Sensor-A', 'Sensor-B']].values.ravel())
    n = len(sensors)
    sensor_to_idx = {s: i for i, s in enumerate(sensors)}
    dist_matrix = np.zeros((n, n))
    for _, row in df.iterrows():
        i, j = sensor_to_idx[row['Sensor-A']], sensor_to_idx[row['Sensor-B']]
        dist_matrix[i, j] = row['Distance']
        dist_matrix[j, i] = row['Distance']
    return df, sensors, dist_matrix


def map_sensors(dist_matrix, sensors, grid_size=20):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(dist_matrix)
    scaler = MinMaxScaler()
    grid_pos = scaler.fit_transform(positions)
    grid_idx = (grid_pos * (grid_size-1)).astype(int)
    return {s: (x, y) for s, (x, y) in zip(sensors, grid_idx)}


def create_synthetic_data(sensors, n_timesteps=100, start_time=None):
    if start_time is None:
        start_time = datetime(2025, 1, 1, 6, 0)
    timestamps = [start_time + timedelta(minutes=5*i)
                  for i in range(n_timesteps)]
    np.random.seed(42)
    data = {'timestamp': timestamps}
    for s in sensors:
        data[int(s)] = np.random.uniform(30, 70, size=n_timesteps)
    return pd.DataFrame(data)


def create_heatmap(sensor_values, sensor_grid_positions, grid_size=20):
    heatmap = np.zeros((grid_size, grid_size))
    for sid, val in sensor_values.items():
        if sid in sensor_grid_positions:
            x, y = sensor_grid_positions[sid]
            heatmap[x, y] = val
    return heatmap


def build_sequences(df, sensor_grid_positions, grid_size=20, seq_len=6):
    X, Y = [], []
    for i in range(len(df) - seq_len):
        seq_maps = []
        for j in range(seq_len):
            vals = df.iloc[i+j].drop(labels='timestamp').to_dict()
            seq_maps.append(create_heatmap(
                vals, sensor_grid_positions, grid_size))
        X.append(np.stack(seq_maps))
        nxt_vals = df.iloc[i+seq_len].drop(labels='timestamp').to_dict()
        Y.append(create_heatmap(nxt_vals, sensor_grid_positions, grid_size))
    return np.array(X), np.array(Y)
