"""Exact COCO-17 graph topology and partition orientation used by PYSKL."""
import torch

# These are PYSKL's original inward links for layout='coco'.
# COCO order: nose, left_eye, right_eye, left_ear, right_ear,
# left_shoulder, right_shoulder, left_elbow, right_elbow,
# left_wrist, right_wrist, left_hip, right_hip,
# left_knee, right_knee, left_ankle, right_ankle.
INWARD = (
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 5), (12, 6), (9, 7), (7, 5),
    (10, 8), (8, 6), (5, 0), (6, 0),
    (1, 0), (3, 1), (2, 0), (4, 2),
)


def _edge2mat(edges, num_nodes):
    matrix = torch.zeros(num_nodes, num_nodes)
    for source, target in edges:
        matrix[target, source] = 1.0
    return matrix


def _normalize_digraph(matrix):
    degree = matrix.sum(0)
    inverse = torch.zeros_like(degree)
    valid = degree > 0
    inverse[valid] = degree[valid].reciprocal()
    return matrix @ torch.diag(inverse)


def adjacency():
    num_nodes = 17
    self_links = tuple((i, i) for i in range(num_nodes))
    outward = tuple((target, source) for source, target in INWARD)
    identity = _edge2mat(self_links, num_nodes)
    inward_matrix = _normalize_digraph(_edge2mat(INWARD, num_nodes))
    outward_matrix = _normalize_digraph(_edge2mat(outward, num_nodes))
    return torch.stack((identity, inward_matrix, outward_matrix))
