"""Standalone implementation of PYSKL's ST-GCN++ for COCO-17 skeletons.

Architecture follows the official PYSKL STGCN backbone:
  graph_cfg = dict(layout='coco', mode='spatial')
  gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn'
  base_channels=64, num_stages=10, ch_ratio=2
  inflate_stages=(5, 8), down_stages=(5, 8)
  data_bn_type='VC'

Input to forward:
  [N, T, 17, 3]       one person (x, y, confidence)
  [N, M, T, 17, 3]    multiple people

The module names backbone.* and cls_head.fc_cls are kept compatible
with the official PYSKL ST-GCN++ state-dict naming.
"""
import torch
import torch.nn as nn

from .graph.coco import adjacency


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
    """PYSKL mstcn: four dilated branches, max-pool branch, and 1x1 branch."""
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 dropout=0.0,
                 ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"),
                 stride=1):
        super().__init__()
        self.ms_cfg = ms_cfg
        branches_count = len(ms_cfg)

        if mid_channels is None:
            mid_channels = out_channels // branches_count
            rem_mid_channels = out_channels - mid_channels * (branches_count - 1)
        else:
            if not isinstance(mid_channels, float) or mid_channels <= 0:
                raise ValueError("mid_channels must be a positive float")
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels
        self.act = nn.ReLU()
        branches = []

        for i, cfg in enumerate(ms_cfg):
            branch_channels = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                branches.append(nn.Conv2d(
                    in_channels, branch_channels, kernel_size=1,
                    stride=(stride, 1)))
            elif cfg[0] == "max":
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                    nn.BatchNorm2d(branch_channels),
                    self.act,
                    nn.MaxPool2d(
                        kernel_size=(cfg[1], 1), stride=(stride, 1),
                        padding=(1, 0))))
            else:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                    nn.BatchNorm2d(branch_channels),
                    self.act,
                    UnitTCN(
                        branch_channels, branch_channels,
                        kernel_size=cfg[0], stride=stride,
                        dilation=cfg[1], norm=False)))

        self.branches = nn.ModuleList(branches)
        total_channels = mid_channels * (branches_count - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(total_channels),
            self.act,
            nn.Conv2d(total_channels, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.transform(x)
        return self.drop(self.bn(x))


class UnitGCN(nn.Module):
    """PYSKL unit_gcn with adaptive='init', conv_pos='pre', and residual."""
    def __init__(self, in_channels, out_channels, A,
                 adaptive="init", with_res=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        if adaptive not in (None, "init", "offset", "importance"):
            raise ValueError(f"Unsupported adaptive mode: {adaptive}")
        self.adaptive = adaptive
        self.with_res = with_res

        if adaptive == "init":
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer("A", A)

        if adaptive in ("offset", "importance"):
            self.PA = nn.Parameter(A.clone())
            if adaptive == "offset":
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            else:
                nn.init.constant_(self.PA, 1)

        self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        if with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels))
            else:
                self.down = nn.Identity()

    def forward(self, x):
        n, _, t, v = x.shape
        residual = self.down(x) if self.with_res else 0

        if self.adaptive in (None, "init"):
            A = self.A
        elif self.adaptive == "offset":
            A = self.A + self.PA
        else:
            A = self.A * self.PA

        x = self.conv(x)
        x = x.view(n, self.num_subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, A).contiguous()
        return self.act(self.bn(x) + residual)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1,
                 residual=True, gcn_adaptive="init", gcn_with_res=True,
                 tcn_dropout=0.0):
        super().__init__()
        self.gcn = UnitGCN(
            in_channels, out_channels, A,
            adaptive=gcn_adaptive, with_res=gcn_with_res)
        self.tcn = MultiScaleTCN(
            out_channels, out_channels, dropout=tcn_dropout, stride=stride)

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(
                in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else 0
        return self.relu(self.tcn(self.gcn(x)) + residual)


class STGCNBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, data_bn_type="VC",
                 ch_ratio=2, num_person=2, num_stages=10,
                 inflate_stages=(5, 8), down_stages=(5, 8),
                 gcn_adaptive="init", gcn_with_res=True, tcn_dropout=0.0):
        super().__init__()
        A = adjacency()
        self.data_bn_type = data_bn_type

        if data_bn_type == "MVC":
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == "VC":
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        elif data_bn_type == "None":
            self.data_bn = nn.Identity()
        else:
            raise ValueError(f"Unsupported data_bn_type: {data_bn_type}")

        modules = []
        if in_channels != base_channels:
            modules.append(STGCNBlock(
                in_channels, base_channels, A.clone(), residual=False,
                gcn_adaptive=gcn_adaptive, gcn_with_res=gcn_with_res))

        inflate_times = 0
        current_channels = base_channels
        for stage in range(2, num_stages + 1):
            stride = 1 + int(stage in down_stages)
            if stage in inflate_stages:
                inflate_times += 1
            out_channels = int(base_channels * ch_ratio ** inflate_times + 1e-4)
            modules.append(STGCNBlock(
                current_channels, out_channels, A.clone(), stride=stride,
                residual=True, gcn_adaptive=gcn_adaptive,
                gcn_with_res=gcn_with_res, tcn_dropout=tcn_dropout))
            current_channels = out_channels

        self.gcn = nn.ModuleList(modules)

    def forward(self, x):
        # Official PYSKL layout: N, M, T, V, C
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
    def __init__(self, num_classes=60, in_channels=256, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.in_c = in_channels
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        n, m, c, t, v = x.shape
        x = x.reshape(n * m, c, t, v)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.reshape(n, m, c).mean(dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)


class STGCNPP(nn.Module):
    """PYSKL ST-GCN++ configured for 2D COCO-17 Joint Motion."""
    def __init__(self, num_classes=60, in_channels=3, num_person=2,
                 num_point=17, base_channels=64, num_stages=10,
                 data_bn_type="VC", ch_ratio=2,
                 inflate_stages=(5, 8), down_stages=(5, 8),
                 gcn_adaptive="init", gcn_with_res=True, tcn_dropout=0.0):
        super().__init__()
        if num_point != 17:
            raise ValueError("This model configuration requires COCO-17 (17 joints).")
        if in_channels != 3:
            raise ValueError("2D Joint Motion input must have x, y, confidence channels.")

        self.backbone = STGCNBackbone(
            in_channels=in_channels, base_channels=base_channels,
            data_bn_type=data_bn_type, ch_ratio=ch_ratio,
            num_person=num_person, num_stages=num_stages,
            inflate_stages=inflate_stages, down_stages=down_stages,
            gcn_adaptive=gcn_adaptive, gcn_with_res=gcn_with_res,
            tcn_dropout=tcn_dropout)
        self.cls_head = GCNHead(
            num_classes=num_classes, in_channels=base_channels * 4)

    def forward(self, keypoint):
        if keypoint.ndim == 4:
            keypoint = keypoint.unsqueeze(1)
        elif keypoint.ndim != 5:
            raise ValueError("Expected [N,T,17,3] or [N,M,T,17,3].")

        if keypoint.shape[-2:] != (17, 3):
            raise ValueError(
                f"Expected 17 joints and 3 channels (x,y,score); got {keypoint.shape[-2:]}.")
        if keypoint.shape[1] < 2:
            padding = keypoint.new_zeros(
                keypoint.shape[0], 2 - keypoint.shape[1],
                keypoint.shape[2], keypoint.shape[3], keypoint.shape[4])
            keypoint = torch.cat((keypoint, padding), dim=1)
        elif keypoint.shape[1] > 2:
            keypoint = keypoint[:, :2]

        return self.cls_head(self.backbone(keypoint))

    def load_pyskl_checkpoint(self, checkpoint_path, strict=True,
                              map_location="cpu", load_head=True):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        cleaned = {key.removeprefix("module."): value for key, value in state.items()}
        if not load_head:
            cleaned = {k: v for k, v in cleaned.items() if not k.startswith("cls_head.")}
        return self.load_state_dict(
            cleaned, strict=(strict if load_head else False))


def prepare_joint_motion(keypoints, image_size, threshold=0.01):
    """PYSKL PreNormalize2D(fix) + GenSkeFeat(dataset='coco', feats=['jm']).

    keypoints: [N,T,17,3] or [N,M,T,17,3], pixel x/y plus confidence.
    image_size: (height, width), as reported by the source video.
    """
    if keypoints.ndim not in (4, 5) or keypoints.shape[-2:] != (17, 3):
        raise ValueError("Expected [N,T,17,3] or [N,M,T,17,3].")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size must contain positive (height, width).")

    joints = keypoints.clone().float()
    low_score = joints[..., 2] <= threshold
    joints[..., 0] = (joints[..., 0] - width / 2) / (width / 2)
    joints[..., 1] = (joints[..., 1] - height / 2) / (height / 2)
    joints[..., 0][low_score] = 0
    joints[..., 1][low_score] = 0

    motion = torch.zeros_like(joints)
    if joints.ndim == 4:
        motion[:, :-1] = joints[:, 1:] - joints[:, :-1]
        motion[:, :-1, :, 2] = (
            joints[:, :-1, :, 2] + joints[:, 1:, :, 2]) / 2
    else:
        motion[:, :, :-1] = joints[:, :, 1:] - joints[:, :, :-1]
        motion[:, :, :-1, :, 2] = (
            joints[:, :, :-1, :, 2] + joints[:, :, 1:, :, 2]) / 2
    return motion
