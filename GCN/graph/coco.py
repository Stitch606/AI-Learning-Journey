"""COCO-17 skeleton topology used by YOLO Pose."""
import torch

# COCO order: nose, left_eye, right_eye, left_ear, right_ear,
# left_shoulder, right_shoulder, left_elbow, right_elbow,
# left_wrist, right_wrist, left_hip, right_hip,
# left_knee, right_knee, left_ankle, right_ankle.
INWARD_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def adjacency():
    v = 17
    identity = torch.eye(v)
    inward = torch.zeros(v, v)
    outward = torch.zeros(v, v)
    for source, target in INWARD_EDGES:
        inward[target, source] = 1.0
        outward[source, target] = 1.0
    def normalize(a):
        degree = a.sum(0)
        inv = torch.zeros_like(degree)
        valid = degree > 0
        inv[valid] = degree[valid].reciprocal()
        return a @ torch.diag(inv)
    return torch.stack((identity, normalize(inward), normalize(outward)))
