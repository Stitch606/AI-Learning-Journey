"""Standalone PyTorch port of PYSKL's ST-GCN++ (JM, COCO-17) tensor architecture.

Runtime dependencies: PyTorch and NumPy only. No PYSKL, MMCV, MMAction,
MMEngine, registry, config parser, or upstream runtime imports.
Module names intentionally mirror the published checkpoint state_dict.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO17_INWARD = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
    (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
    (1, 0), (3, 1), (2, 0), (4, 2),
)


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 dilation=1, norm=True, dropout=0.0):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, 1),
            padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class MultiScaleTCN(nn.Module):
    """PYSKL mstcn with the same branch/module registration order."""
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 dropout=0.0,
                 ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"),
                 stride=1):
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()
        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels
        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels
        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1,
                                          stride=(stride, 1)))
            elif cfg[0] == "max":
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c), self.act,
                    nn.MaxPool2d(kernel_size=(cfg[1], 1),
                                 stride=(stride, 1), padding=(1, 0))))
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c), self.act,
                    UnitTCN(branch_c, branch_c, kernel_size=cfg[0],
                            stride=stride, dilation=cfg[1], norm=False))
                branches.append(branch)
        self.branches = nn.ModuleList(branches)
        total_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(total_channels), self.act,
            nn.Conv2d(total_channels, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        out = self.transform(out)
        return self.drop(self.bn(out))


class UnitGCN(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency,
                 adaptive="init", conv_pos="pre", with_res=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = adjacency.size(0)
        self.adaptive = adaptive
        self.conv_pos = conv_pos
        self.with_res = with_res
        if adaptive == "init":
            self.A = nn.Parameter(adjacency.clone())
        else:
            self.register_buffer("A", adjacency)
        if conv_pos == "pre":
            self.conv = nn.Conv2d(in_channels, out_channels * adjacency.size(0), 1)
        else:
            self.conv = nn.Conv2d(adjacency.size(0) * in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        if with_res and in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        elif with_res:
            self.down = lambda x: x

    def forward(self, x, A=None):
        n, c, t, v = x.shape
        residual = self.down(x) if self.with_res else 0
        a = self.A
        if self.conv_pos == "pre":
            x = self.conv(x).view(n, self.num_subsets, -1, t, v)
            x = torch.einsum("nkctv,kvw->nctw", x, a).contiguous()
        else:
            x = torch.einsum("nctv,kvw->nkctw", x, a).contiguous()
            x = self.conv(x.view(n, -1, t, v))
        return self.act(self.bn(x) + residual)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1,
                 residual=True):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, adjacency,
                           adaptive="init", with_res=True)
        self.tcn = MultiScaleTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels,
                                    kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        res = self.residual(x)
        return self.relu(self.tcn(self.gcn(x, A)) + res)


class STGCNBackbone(nn.Module):
    @staticmethod
    def _edge2mat(edges, n):
        a = np.zeros((n, n), dtype=np.float32)
        for source, target in edges:
            a[target, source] = 1.0
        return a

    @staticmethod
    def _normalize_digraph(a):
        degree = a.sum(axis=0)
        inv = np.zeros_like(degree)
        valid = degree > 0
        inv[valid] = 1.0 / degree[valid]
        return a @ np.diag(inv)

    @classmethod
    def coco_spatial_adjacency(cls):
        n = 17
        self_links = tuple((i, i) for i in range(n))
        outward = tuple((j, i) for i, j in COCO17_INWARD)
        return np.stack((
            cls._edge2mat(self_links, n),
            cls._normalize_digraph(cls._edge2mat(COCO17_INWARD, n)),
            cls._normalize_digraph(cls._edge2mat(outward, n)),
        )).astype(np.float32)

    def __init__(self, in_channels=3, base_channels=64, data_bn_type="VC",
                 ch_ratio=2, num_person=2, num_stages=10,
                 inflate_stages=(5, 8), down_stages=(5, 8)):
        super().__init__()
        adjacency = torch.tensor(self.coco_spatial_adjacency(), dtype=torch.float32)
        self.data_bn_type = data_bn_type
        if data_bn_type == "MVC":
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * adjacency.size(1))
        elif data_bn_type == "VC":
            self.data_bn = nn.BatchNorm1d(in_channels * adjacency.size(1))
        else:
            self.data_bn = nn.Identity()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = tuple(inflate_stages)
        self.down_stages = tuple(down_stages)
        modules = []
        if in_channels != base_channels:
            modules.append(STGCNBlock(in_channels, base_channels,
                                      adjacency.clone(), 1, residual=False))
        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in self.down_stages)
            stage_in = base_channels
            if i in self.inflate_stages:
                inflate_times += 1
            out_channels = int(64 * self.ch_ratio ** inflate_times + 1e-4)
            base_channels = out_channels
            modules.append(STGCNBlock(stage_in, out_channels,
                                      adjacency.clone(), stride))
        if in_channels == base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)

    def forward(self, x):
        # Input: N, M, T, V, C
        n, m, t, v, c = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == "MVC":
            x = self.data_bn(x.view(n, m * v * c, t))
        else:
            x = self.data_bn(x.view(n * m, v * c, t))
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)
        for block in self.gcn:
            x = block(x)
        return x.reshape((n, m) + x.shape[1:])


class GCNHead(nn.Module):
    def __init__(self, num_classes=60, in_channels=256):
        super().__init__()
        self.in_c = in_channels
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # Backbone output: N, M, C, T, V
        n, m, c, t, v = x.shape
        x = x.reshape(n * m, c, t, v)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.reshape(n, m, c).mean(dim=1)
        return self.fc_cls(x)


class STGCNPP(nn.Module):
    """Single public interface for the standalone ST-GCN++ model.

    The internal backbone/head classes preserve the checkpoint module names.
    Raw YOLO COCO-17 keypoints can be passed directly to predict().
    """
    def __init__(
        self,
        checkpoint_path=None,
        num_classes=60,
        device="cpu",
        strict=True,
    ):
        super().__init__()
        self.backbone = STGCNBackbone()
        self.cls_head = GCNHead(num_classes=num_classes, in_channels=256)
        self.device = torch.device(device)

        if checkpoint_path is not None:
            self.load_weights(checkpoint_path, strict=strict)

        self.to(self.device)
        self.eval()

    def load_weights(self, checkpoint_path, strict=True):
        """Load a PYSKL-compatible checkpoint without changing state_dict names."""
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise TypeError("Checkpoint does not contain a state_dict mapping")
        if state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        return self.load_state_dict(state, strict=strict)

    @staticmethod
    def validate_keypoints(keypoints):
        """Validate raw YOLO COCO-17 data with shape [T,17,3]."""
        arr = np.asarray(keypoints, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (17, 3):
            raise ValueError(
                f"Expected [T,17,3] (x,y,confidence), got {arr.shape}"
            )
        if arr.shape[0] < 1:
            raise ValueError("At least one frame is required")
        if not np.isfinite(arr).all():
            raise ValueError("Keypoints contain NaN or infinite values")
        if np.any(arr[..., 2] < 0) or np.any(arr[..., 2] > 1):
            raise ValueError("Confidence values must be in [0,1]")
        return arr

    @classmethod
    def prepare_input(cls, keypoints, image_size=(1920, 1080), clip_len=100):
        """Convert raw YOLO COCO-17 coordinates into packed JM model input."""
        arr = cls.validate_keypoints(keypoints)
        if clip_len != 100:
            raise ValueError("This checkpoint expects exactly 100 frames")

        indices = np.linspace(0, len(arr) - 1, clip_len).round().astype(np.int64)
        sampled = arr[indices].copy()

        # PYSKL PreNormalize2D default mode='fix', img_shape=(height,width).
        h, w = int(image_size[1]), int(image_size[0])
        xy = sampled[..., :2]
        score = sampled[..., 2]
        mask = score > 0.01
        xy[..., 0] = (xy[..., 0] - (w / 2.0)) / (w / 2.0)
        xy[..., 1] = (xy[..., 1] - (h / 2.0)) / (h / 2.0)
        xy[~mask] = 0.0

        # PYSKL ToMotion(dataset='coco'): frame differences and adjacent
        # confidence mean; the final frame remains zero.
        jm = np.zeros_like(sampled, dtype=np.float32)
        jm[:-1, :, :2] = np.diff(xy, axis=0)
        jm[:-1, :, 2] = (score[:-1] + score[1:]) / 2.0

        packed = np.zeros((1, 1, 2, clip_len, 17, 3), dtype=np.float32)
        packed[0, 0, 0] = jm
        return packed

    def predict(self, keypoints, image_size=(1920, 1080)):
        """Run end-to-end inference from raw [T,17,3] YOLO keypoints."""
        packed = self.prepare_input(keypoints, image_size=image_size)
        tensor = torch.from_numpy(packed).to(self.device)
        self.eval()
        with torch.inference_mode():
            return self(tensor)

    def forward_logits(self, keypoint):
        if keypoint.ndim != 6:
            raise ValueError("Expected [B, clips, persons, T, V, C] input")
        b, clips = keypoint.shape[:2]
        x = keypoint.reshape((b * clips,) + keypoint.shape[2:])
        features = self.backbone(x)
        logits = self.cls_head(features)
        return logits.reshape(b, clips, -1).mean(dim=1)

    def forward(self, keypoint, return_loss=False, **kwargs):
        logits = self.forward_logits(keypoint)
        # PYSKL RecognizerGCN defaults to average_clips='prob'.
        return torch.softmax(logits, dim=-1)
