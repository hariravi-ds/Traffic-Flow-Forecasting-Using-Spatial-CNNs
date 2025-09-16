import os
from tensorflow.keras.models import load_model


def save_model(model, path="models/cnn_model.h5"):
    model.save(path)


def load_saved_model(path="models/cnn_model.h5"):
    return load_model(path)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
