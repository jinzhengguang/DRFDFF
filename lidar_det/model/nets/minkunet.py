"""
From https://github.com/mit-han-lab/spvnas/blob/master/core/models/semantic_kitti/minkunet.py
https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/minkunet.py
"""

import time
from collections import OrderedDict
import numpy as np
import torch
import torchsparse
import torch.nn as nn
import torchsparse.nn as spnn
import MinkowskiEngine as ME

__all__ = ['MinkUNet']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transpose=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ADFF(nn.Module):
    def __init__(self, inc, outc, ks=2, stride=2, dilation=1):
        super().__init__()
        self.scbrk2 = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
        )
        self.scbrk3 = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=3, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
        )
        self.relu = spnn.ReLU(True)
        self.scbrkw = nn.Parameter(torch.randn(2))
        # self.scbrkw = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # 2024-04-09 Jinzheng Guang
        gates = self.sigmoid(self.scbrkw)
        outputs = self.scbrk2(inputs)
        outputs.F = gates[0] * outputs.F + gates[1] * self.scbrk3(inputs).F
        return self.relu(outputs)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        # Global coords does not require coords_key
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid()
        )
        # self.pooling = ME.MinkowskiGlobalPooling()
        self.pooling = ME.MinkowskiGlobalMaxPooling()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, inputs):
        # 2024-03-31 Jinzheng Guang
        feat = inputs.F
        coords = inputs.C
        stride = inputs.s

        x = ME.SparseTensor(feat, coords)
        y = self.pooling(x)
        y = self.fc(y)
        y = self.broadcast_mul(x, y)

        outputs = torchsparse.SparseTensor(feats=y.F, coords=coords, stride=stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps

        return outputs


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 2024-03-26 Jinzheng Guang CBAM attention
        self.spnnavg = spnn.GlobalAveragePooling()
        self.spnnmax = spnn.GlobalMaxPooling()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # 2024-03-26 Jinzheng Guang CBAM attention
        feats = inputs.F
        coords = inputs.C
        stride = inputs.s

        # feats = super().forward(feats)
        avgp = self.spnnavg(inputs)
        maxp = self.spnnmax(inputs)
        atts = self.sigmoid(self.fc(avgp) + self.fc(maxp))
        # output = feats * atts

        batch_index = coords[:, -1]
        max_index = torch.max(batch_index).item()
        output = []
        for i in range(max_index + 1):
            cur_inputs = torch.index_select(feats, 0, torch.where(batch_index == i)[0])
            att = atts[i].unsqueeze(0)
            cur_outputs = cur_inputs * att
            output.append(cur_outputs)
        outputcat = torch.cat(output, 0)

        outputs = torchsparse.SparseTensor(coords=coords, feats=outputcat, stride=stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps

        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc)
        )
        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class ResidualBlockgjz(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc)
        )
        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )
        self.relu = spnn.ReLU(True)
        self.catt = ksatt(channel=outc)

    def forward(self, x):
        out = self.relu(self.catt(self.net(x)) + self.downsample(x))
        return out


class Paths(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.basicconv = BasicConvolutionBlock(inc, outc)
        self.catt = ChannelAttention(channel=outc)
        # self.catt = SELayer(outc)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.catt(self.basicconv(x)) + x)
        return out


class ksatt(nn.Module):
    def __init__(self, channel, kernel_size=3, dilation=1, stride=1, nx=3):
        super().__init__()
        # 2024-04-09 Jinzheng Guang conv ks attention
        self.channel = channel
        self.nx = nx
        self.Conv3d = nn.ModuleList([])
        for i in range(nx):
            self.Conv3d.append(BasicConvolutionBlock(channel, channel, ks=kernel_size, dilation=dilation, stride=stride))
        self.catt = ChannelAttention(channel=channel * nx)
        self.weight = nn.Parameter(torch.randn(nx))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 2024-04-05 Jinzheng Guang
        feats = x.F
        coords = x.C
        stride = x.s
        coord_maps = x.coord_maps
        kernel_maps = x.kernel_maps

        kx = []
        for conv in self.Conv3d:
            x = conv(x)
            kx.append(x)
        xcat = torchsparse.cat(kx)

        outatt = self.catt(xcat)
        outnx = outatt.F.view(-1, self.channel, self.nx)

        weights = self.sigmoid(self.weight)
        outf = []
        for ix in range(self.nx):
            outf.append((kx[ix].F + outnx[:,:,ix]) * weights[ix])
        output = sum(outf)

        outputs = torchsparse.SparseTensor(coords=coords, feats=output, stride=stride)
        outputs.coord_maps = coord_maps
        outputs.kernel_maps = kernel_maps

        return outputs


class MinkUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        input_dim = kwargs.get("input_dim", 3)

        self.stem = nn.Sequential(
            spnn.Conv3d(input_dim, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            # BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ADFF(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlockgjz(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            ADFF(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlockgjz(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            ADFF(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlockgjz(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            ADFF(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlockgjz(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        # 2023-03-31 Jinzheng Guang
        # cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        self.pathx0 = nn.Sequential(
            Paths(cs[0], cs[0]),
            Paths(cs[0], cs[0]),
            Paths(cs[0], cs[0]),
            Paths(cs[0], cs[0])
        )
        self.pathx1 = nn.Sequential(
            Paths(cs[1], cs[1]),
            Paths(cs[1], cs[1]),
            Paths(cs[1], cs[1])
        )
        self.pathx2 = nn.Sequential(
            Paths(cs[2], cs[2]),
            Paths(cs[2], cs[2])
        )
        self.pathx3 = nn.Sequential(
            Paths(cs[3], cs[3])
        )

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, spnn.Conv3d):
                # ME 的何凯明初始化
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, spnn.BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                pass

    def forward(self, x):
        x0 = self.stem(x)  # 621,687 x 3
        x1 = self.stage1(x0)  # 621,687 x 32
        x2 = self.stage2(x1)  # 362,687 x 32
        x3 = self.stage3(x2)  # 192,434 x 64
        x4 = self.stage4(x3)  # 94,584 x 128

        y1 = self.up1[0](x4)  # 42,187 x 256
        y1 = torchsparse.cat([y1, self.pathx3(x3)])
        y1 = self.up1[1](y1)  # 94,584 x 256

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, self.pathx2(x2)])
        y2 = self.up2[1](y2)  # 192,434 x 128

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, self.pathx1(x1)])
        y3 = self.up3[1](y3)  # 362,687 x 96

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, self.pathx0(x0)])
        y4 = self.up4[1](y4)  # 621,687 x 96

        out = self.classifier(y4.F)  # 621,687 x 31

        return out  # (n, 31)
