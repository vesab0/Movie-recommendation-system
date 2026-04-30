import math

def euclidean_distance(v1, v2):
    """
    Straight-line distance between two vectors.
    Good for dense numerical vectors.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def manhattan_distance(v1, v2):
    """
    Sum of absolute differences. More robust to outliers than Euclidean.
    """
    return sum(abs(a - b) for a, b in zip(v1, v2))


def cosine_similarity(v1, v2):
    """
    Measures the angle between vectors, ignoring magnitude.
    Range: -1 to 1. Excellent for sparse multi-hot features (genres, cast, etc.)
    We return similarity (higher = more similar), not distance.
    """
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0  # One vector is all zeros
    
    return dot_product / (norm1 * norm2)


def cosine_distance(v1, v2):
    """
    Convert cosine similarity to a distance metric (0 = identical, 2 = opposite).
    """
    return 1.0 - cosine_similarity(v1, v2)
