import numpy as np


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


def cosine_distance(v1, v2):
    return 1.0 - cosine_similarity(v1, v2)