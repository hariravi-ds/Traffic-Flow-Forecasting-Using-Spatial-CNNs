import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt


def build_model(hp):
    model = keras.Sequential([
        layers.Input(shape=(6, 20, 20, 1)),
        layers.Conv3D(filters=hp.Choice('filters_1', [16, 32, 64]), kernel_size=(
            2, 3, 3), activation='relu', padding='same'),
        layers.Conv3D(filters=hp.Choice('filters_2', [16, 32, 64]), kernel_size=(
            2, 3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(400, activation='relu'),
        layers.Dense(400, activation='linear')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
        loss='mse', metrics=['mae']
    )
    return model
