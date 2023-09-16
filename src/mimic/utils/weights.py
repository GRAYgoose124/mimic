import numpy as np


def normalize_weights(weights):
    norm = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized_weights = weights / norm
    return normalized_weights


def total_norm(weights):
    for i, w in enumerate(weights):
        weights[i] = w / np.linalg.norm(w)

    return weights
