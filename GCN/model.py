"""PYSKL-compatible ST-GCN++ for COCO-17 Joint-Motion action recognition.

This is a standalone PyTorch port of the official PYSKL ST-GCN++ components
used by configs/stgcn++/stgcn++_ntu60_xsub_hrnet/jm.py.

Checkpoint-compatible module names:
    backbone.data_bn
    backbone.gcn.*
    cls_head.fc_cls

Input accepted:
    [N, T, V, C]       one person
    [N, M, T, V, C]    M people
where V=17 and C=3 (x, y, confidence) for the published 2D JM checkpoint.
"""
import torch
import torch.nn as nn

from .graph.coco import adjacency as coco_adjacency


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 dilation=1, norm=True, dropout=0.0):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
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
        num_branches = len(ms_cfg)

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
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
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                branches.append(nn.Conv2d(
                    in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue

            if cfg[0] == "max":
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c),
                    self.act,
                    nn.MaxPool2d(
                        kernel_size=(cfg[1], 1),
                        stride=(stride, 1),
                        padding=(1, 0))))
                continue

            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c),
                self.act,
                UnitTCN(
                    branch_c, branch_c,
                    kernel_size=cfg[0],
                    stride=stride,
                    dilation=cfg[1],
                    norm=False))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        total_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(total_channels),
            self.act,
            nn.Conv2d(total_channels, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        x = self.transform(x)
        return self.drop(self.bn(x))


class UnitGCN(nn.Module):
    """PYSKL unit_gcn with adaptive='init' and optional residual connection."""
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

        # PYSKL's default conv_pos='pre'.
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
                 tcn_type="mstcn", tcn_dropout=0.0):
        super().__init__()
        self.gcn = UnitGCN(
            in_channels, out_channels, A,
            adaptive=gcn_adaptive, with_res=gcn_with_res)

        if tcn_type == "mstcn":
            self.tcn = MultiScaleTCN(
                out_channels, out_channels, dropout=tcn_dropout)
        elif tcn_type == "unit_tcn":
            self.tcn = UnitTCN(
                out_channels, out_channels, kernel_size=9,
                stride=stride, dropout=tcn_dropout)
        else:
            raise ValueError(f"Unsupported TCN type: {tcn_type}")

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
    """PYSKL STGCN backbone configured exactly as the ST-GCN++ JM config."""
    def __init__(self, in_channels=3, base_channels=64, data_bn_type="VC",
                 ch_ratio=2, num_person=2, num_stages=10,
                 inflate_stages=(5, 8), down_stages=(5, 8),
                 gcn_adaptive="init", gcn_with_res=True,
                 tcn_type="mstcn", tcn_dropout=0.0):
        super().__init__()
        A = coco_adjacency()
        self.data_bn_type = data_bn_type

        if data_bn_type == "MVC":
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == "VC":
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        elif data_bn_type == "None":
            self.data_bn = nn.Identity()
        else:
            raise ValueError(f"Unsupported data_bn_type: {data_bn_type}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = tuple(inflate_stages)
        self.down_stages = tuple(down_stages)

        modules = []
        if in_channels != base_channels:
            modules.append(STGCNBlock(
                in_channels, base_channels, A.clone(),
                residual=False,
                gcn_adaptive=gcn_adaptive,
                gcn_with_res=gcn_with_res,
                tcn_type=tcn_type,
                tcn_dropout=0.0))

        inflate_times = 0
        current_base = base_channels
        for stage in range(2, num_stages + 1):
            stride = 1 + int(stage in self.down_stages)
            in_stage = current_base
            if stage in self.inflate_stages:
                inflate_times += 1
            out_stage = int(self.base_channels * self.ch_ratio ** inflate_times + 1e-4)
            modules.append(STGCNBlock(
                in_stage, out_stage, A.clone(),
                stride=stride,
                residual=True,
                gcn_adaptive=gcn_adaptive,
                gcn_with_res=gcn_with_res,
                tcn_type=tcn_type,
                tcn_dropout=tcn_dropout))
            current_base = out_stage

        self.num_stages = len(modules)
        self.gcn = nn.ModuleList(modules)

    def forward(self, x):
        # PYSKL layout: N, M, T, V, C
        n, m, t, v, c = x.size()
        if self.data_bn_type == "MVC":
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = self.data_bn(x.view(n, m * v * c, t))
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = self.data_bn(x.view(n * m, v * c, t))

        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        for block in self.gcn:
            x = block(x)

        return x.reshape((n, m) + x.shape[1:])


class GCNHead(nn.Module):
    """PYSKL GCNHead (mode='GCN', dropout=0)."""
    def __init__(self, num_classes=60, in_channels=256, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.in_c = in_channels
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: N, M, C, T, V
        n, m, c, t, v = x.shape
        x = x.reshape(n * m, c, t, v)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.reshape(n, m, c).mean(dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)


class STGCNPP(nn.Module):
    """PYSKL-compatible ST-GCN++ recognizer with the official JM checkpoint layout."""
    def __init__(self, num_classes=60, in_channels=3, num_person=2,
                 num_point=17, base_channels=64, num_stages=10,
                 data_bn_type="VC", ch_ratio=2,
                 inflate_stages=(5, 8), down_stages=(5, 8),
                 gcn_adaptive="init", gcn_with_res=True,
                 tcn_type="mstcn", tcn_dropout=0.0):
        super().__init__()
        if num_point != 17:
            raise ValueError("This pretrained configuration is for COCO-17 (17 joints).")
        if in_channels != 3:
            raise ValueError("The published 2D JM checkpoint expects x, y, confidence (3 channels).")

        self.backbone = STGCNBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            data_bn_type=data_bn_type,
            ch_ratio=ch_ratio,
            num_person=num_person,
            num_stages=num_stages,
            inflate_stages=inflate_stages,
            down_stages=down_stages,
            gcn_adaptive=gcn_adaptive,
            gcn_with_res=gcn_with_res,
            tcn_type=tcn_type,
            tcn_dropout=tcn_dropout)
        self.cls_head = GCNHead(num_classes=num_classes, in_channels=base_channels * 4)

    def forward(self, keypoint):
        # Also accept the common single-person [N,T,V,C] layout.
        if keypoint.ndim == 4:
            keypoint = keypoint.unsqueeze(1)
        elif keypoint.ndim == 5:
            # Preserve compatibility with the former [N,C,T,V,M] layout.
            if keypoint.shape[1] == 3 and keypoint.shape[-1] in (1, 2):
                keypoint = keypoint.permute(0, 4, 2, 3, 1).contiguous()
        else:
            raise ValueError("Expected [N,T,V,C] or [N,M,T,V,C].")

        if keypoint.shape[-2] != 17 or keypoint.shape[-1] != 3:
            raise ValueError(
                f"Expected V=17 and C=3 (x,y,confidence); got V={keypoint.shape[-2]}, C={keypoint.shape[-1]}")

        features = self.backbone(keypoint)
        return self.cls_head(features)

    def load_pyskl_checkpoint(self, checkpoint_path, strict=True, map_location="cpu"):
        """Load an official PYSKL checkpoint (e.g. stgcnpp.../jm.pth)."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        cleaned = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            cleaned[key] = value
        return self.load_state_dict(cleaned, strict=strict)
