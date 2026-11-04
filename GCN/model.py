"""ST-GCN++ for skeleton-based action recognition.

Input:
    [N, C, T, V, M] (preferred), or [N, T, V, C] for a single person.
The default graph is COCO-17. This implementation uses PyTorch only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_digraph(a):
    degree = a.sum(0)
    inv = torch.zeros_like(degree)
    mask = degree > 0
    inv[mask] = degree[mask].reciprocal()
    return a @ torch.diag(inv)


def coco17_adjacency():
    """Return the 3-partition COCO-17 adjacency: self, inward, outward."""
    inward_edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    v = 17
    identity = torch.eye(v)
    inward = torch.zeros(v, v)
    outward = torch.zeros(v, v)
    for source, target in inward_edges:
        inward[target, source] = 1.0
        outward[source, target] = 1.0
    return torch.stack((identity, _normalize_digraph(inward),
                        _normalize_digraph(outward)))


class MultiScaleTemporalConv(nn.Module):
    """Parallel temporal branches with multiple dilation rates and pooling."""
    def __init__(self, in_channels, out_channels, stride=1,
                 dilations=(1, 2), dropout=0.0):
        super().__init__()
        branch_count = len(dilations) + 2
        if out_channels % branch_count:
            raise ValueError("out_channels must be divisible by len(dilations)+2")
        branch_channels = out_channels // branch_count
        branches = []
        for dilation in dilations:
            padding = dilation
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_channels, branch_channels,
                          kernel_size=(3, 1), stride=(stride, 1),
                          padding=(padding, 0), dilation=(dilation, 1),
                          bias=False),
                nn.BatchNorm2d(branch_channels)))
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1),
                          padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)))
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1,
                      stride=(stride, 1), bias=False),
            nn.BatchNorm2d(branch_channels)))
        self.branches = nn.ModuleList(branches)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(torch.cat([branch(x) for branch in self.branches], dim=1))


class SpatialGraphConv(nn.Module):
    """Adaptive spatial graph convolution over the three COCO partitions."""
    def __init__(self, in_channels, out_channels, adjacency,
                 adaptive=True, residual=True):
        super().__init__()
        if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("adjacency must have shape [K, V, V]")
        self.num_subsets = adjacency.shape[0]
        self.register_buffer("A", adjacency.float())
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(adjacency.float().clone())
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(self.num_subsets)
        ])
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.bn = nn.BatchNorm2d(out_channels)
        if not residual:
            self.down = None
        elif in_channels == out_channels:
            self.down = nn.Identity()
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        adjacency = self.PA if self.adaptive else self.A
        out = 0
        for k, conv in enumerate(self.convs):
            # [N, C, T, V] x [V, V] -> [N, C, T, V]
            out = out + torch.einsum("nctv,vw->nctw", conv(x),
                                     adjacency[k] + self.alpha[k] * self.A[k])
        out = self.bn(out)
        if self.down is not None:
            out = out + self.down(x)
        return self.relu(out)


class STGCNPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1,
                 residual=True, adaptive=True, dropout=0.0):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, adjacency,
                                    adaptive=adaptive, residual=True)
        self.tcn = MultiScaleTemporalConv(out_channels, out_channels,
                                          stride=stride, dropout=dropout)
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.tcn(self.gcn(x))
        if self.residual is not None:
            y = y + self.residual(x)
        return self.relu(y)


class STGCNPP(nn.Module):
    """ST-GCN++ backbone + global pooling + classification head."""
    def __init__(self, num_classes, in_channels=3, num_person=1,
                 num_point=17, base_channels=64, num_stages=10,
                 adaptive=True, dropout=0.0, graph=None):
        super().__init__()
        if graph is None:
            adjacency = coco17_adjacency()
        else:
            adjacency = torch.as_tensor(graph, dtype=torch.float32)
        if adjacency.shape[-1] != num_point:
            raise ValueError("Graph vertex count must equal num_point")
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.data_bn = nn.BatchNorm1d(num_person * num_point * in_channels)
        channels = [base_channels] * num_stages
        for index in (4, 7):
            if index < num_stages:
                channels[index:] = [c * 2 for c in channels[index:]]
        blocks = []
        current = in_channels
        for index, output in enumerate(channels):
            stride = 2 if index in (4, 7) else 1
            blocks.append(STGCNPPBlock(
                current, output, adjacency, stride=stride,
                residual=index != 0, adaptive=adaptive, dropout=dropout))
            current = output
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(current, num_classes)

    def forward(self, x):
        # Accept [N,T,V,C] for one person or [N,C,T,V,M].
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
        elif x.ndim != 5:
            raise ValueError("Expected [N,T,V,C] or [N,C,T,V,M]")
        n, c, t, v, m = x.shape
        if c != self.in_channels or v != self.num_point:
            raise ValueError(f"Expected C={self.in_channels}, V={self.num_point}; got C={c}, V={v}")
        if m != self.num_person:
            raise ValueError(f"Expected M={self.num_person} people; got M={m}")
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2)
        x = x.contiguous().view(n * m, c, t, v)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=(2, 3)).view(n, m, -1).mean(dim=1)
        return self.head(x)
