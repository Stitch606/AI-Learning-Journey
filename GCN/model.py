import math
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# ============================================================
# 1. Utility Functions
# ============================================================

def import_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")

    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(module):
    classname = module.__class__.__name__

    if "Conv" in classname:
        if hasattr(module, "weight"):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")

        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif "BatchNorm" in classname:
        if module.weight is not None:
            module.weight.data.normal_(1.0, 0.02)

        if module.bias is not None:
            module.bias.data.zero_()


# ============================================================
# 2. Temporal Convolution Blocks
# ============================================================

class TemporalConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1
    ):
        super().__init__()

        padding = (
            kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        ) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilations=(1, 2, 3, 4),
        residual=True,
        residual_kernel_size=1
    ):
        super().__init__()

        assert out_channels % (len(dilations) + 2) == 0

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                )
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=(3, 1),
                    stride=(stride, 1),
                    padding=(1, 0)
                ),
                nn.BatchNorm2d(branch_channels)
            )
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(branch_channels)
            )
        )

        if not residual:
            self.residual = lambda x: 0

        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x

        else:
            self.residual = TemporalConv(
                in_channels,
                out_channels,
                kernel_size=residual_kernel_size,
                stride=stride
            )

        self.apply(weights_init)

    def forward(self, x):
        residual = self.residual(x)

        branch_outputs = [
            branch(x) for branch in self.branches
        ]

        output = torch.cat(branch_outputs, dim=1)
        output += residual

        return output


# ============================================================
# 3. CTRGC: Channel-wise Topology Refinement Graph Convolution
# ============================================================

class CTRGC(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        rel_reduction=8,
        mid_reduction=1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels in (3, 9):
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction

        self.conv1 = nn.Conv2d(
            in_channels,
            self.rel_channels,
            kernel_size=1
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            self.rel_channels,
            kernel_size=1
        )

        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.conv4 = nn.Conv2d(
            self.rel_channels,
            out_channels,
            kernel_size=1
        )

        self.tanh = nn.Tanh()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)

            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)

    def forward(self, x, A=None, alpha=1):
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)

        relation = self.tanh(
            x1.unsqueeze(-1) - x2.unsqueeze(-2)
        )

        relation = self.conv4(relation) * alpha

        if A is not None:
            relation = relation + A.unsqueeze(0).unsqueeze(0)

        output = torch.einsum(
            "ncuv,nctv->nctu",
            relation,
            x3
        )

        return output


# ============================================================
# 4. GCN Block
# ============================================================

class unit_gcn(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        coff_embedding=4,
        adaptive=True,
        residual=True
    ):
        super().__init__()

        self.adaptive = adaptive
        self.num_subset = A.shape[0]

        self.convs = nn.ModuleList([
            CTRGC(in_channels, out_channels)
            for _ in range(self.num_subset)
        ])

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1
                    ),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        if adaptive:
            self.PA = nn.Parameter(
                torch.from_numpy(A.astype(np.float32))
            )
        else:
            self.A = Variable(
                torch.from_numpy(A.astype(np.float32)),
                requires_grad=False
            )

        self.alpha = nn.Parameter(torch.zeros(1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)

            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)

        bn_init(self.bn, 1e-6)

    def forward(self, x):
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())

        output = None

        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            output = z if output is None else output + z

        output = self.bn(output)
        output += self.down(x)
        output = self.relu(output)

        return output


# ============================================================
# 5. Temporal Convolution Unit
# ============================================================

class unit_tcn(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


# ============================================================
# 6. Combined TCN + GCN Unit
# ============================================================

class TCN_GCN_unit(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        residual=True,
        adaptive=True,
        kernel_size=5,
        dilations=(1, 2)
    ):
        super().__init__()

        self.gcn1 = unit_gcn(
            in_channels,
            out_channels,
            A,
            adaptive=adaptive
        )

        self.tcn1 = MultiScale_TemporalConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False
        )

        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride
            )

    def forward(self, x):
        y = self.gcn1(x)
        y = self.tcn1(y)
        y = y + self.residual(x)

        return self.relu(y)


# ============================================================
# 7. Main CTR-GCN Model
# ============================================================

class TCG(nn.Module):

    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        graph=None,
        graph_args=None,
        in_channels=3,
        drop_out=0,
        adaptive=True
    ):
        super().__init__()

        if graph is None:
            raise ValueError("A graph import path must be provided.")

        if graph_args is None:
            graph_args = {}

        Graph = import_class(graph)
        self.graph = Graph(**graph_args)

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point

        self.data_bn = nn.BatchNorm1d(
            num_person * in_channels * num_point
        )

        base_channel = 64

        self.l1 = TCN_GCN_unit(
            in_channels, base_channel, A,
            residual=False, adaptive=adaptive
        )

        self.l2 = TCN_GCN_unit(
            base_channel, base_channel, A,
            adaptive=adaptive
        )

        self.l3 = TCN_GCN_unit(
            base_channel, base_channel, A,
            adaptive=adaptive
        )

        self.l4 = TCN_GCN_unit(
            base_channel, base_channel, A,
            adaptive=adaptive
        )

        self.l5 = TCN_GCN_unit(
            base_channel, base_channel * 2, A,
            stride=2, adaptive=adaptive
        )

        self.l6 = TCN_GCN_unit(
            base_channel * 2, base_channel * 2, A,
            adaptive=adaptive
        )

        self.l7 = TCN_GCN_unit(
            base_channel * 2, base_channel * 2, A,
            adaptive=adaptive
        )

        self.l8 = TCN_GCN_unit(
            base_channel * 2, base_channel * 4, A,
            stride=2, adaptive=adaptive
        )

        self.l9 = TCN_GCN_unit(
            base_channel * 4, base_channel * 4, A,
            adaptive=adaptive
        )

        self.l10 = TCN_GCN_unit(
            base_channel * 4, base_channel * 4, A,
            adaptive=adaptive
        )

        self.fc = nn.Linear(
            base_channel * 4,
            num_class
        )

        nn.init.normal_(
            self.fc.weight,
            0,
            math.sqrt(2.0 / num_class)
        )

        bn_init(self.data_bn, 1)

        self.drop_out = (
            nn.Dropout(drop_out)
            if drop_out
            else nn.Identity()
        )

    def forward(self, x):
        if x.ndim == 3:
            N, T, VC = x.shape

            x = (
                x.view(N, T, self.num_point, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                .unsqueeze(-1)
            )

        N, C, T, V, M = x.size()

        x = (
            x.permute(0, 4, 3, 1, 2)
            .contiguous()
            .view(N, M * V * C, T)
        )

        x = self.data_bn(x)

        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        c_new = x.size(1)

        x = x.view(N, M, c_new, -1)
        x = x.mean(-1).mean(1)

        x = self.drop_out(x)

        return self.fc(x)
