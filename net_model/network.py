import torch
import torch.nn as nn
from net_model.resnet import resnet18, wide_resnet50_2
from net_model.de_resnet import de_resnet18, de_wide_resnet50_2
import random


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x


class TFA(nn.Module):
    def __init__(self, in_channel, ou_channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 2, ou_channel, 1, bias=True),
            CBAM(ou_channel, ratio=16, kernel_size=3),
        )
        if in_channel != ou_channel:
            self.shortcut = nn.Conv2d(in_channel, ou_channel, 1, bias=True)

    def forward(self, input):
        out = self.conv(input)
        if input.shape[1] != out.shape[1]:
            input = self.shortcut(input)
        out += input

        return out


class UniRD(nn.Module):
    def __init__(self, backbone='resnet18', drop_ratio=0.6):
        super(UniRD, self).__init__()

        if backbone == 'resnet18':
            self.encoder, self.bn = resnet18(pretrained=True)
            self.decoder = de_resnet18(pretrained=False)
            ch_list = [64, 128, 256]
        elif backbone == 'wide_resnet50':
            self.encoder, self.bn = wide_resnet50_2(pretrained=True)
            self.decoder = de_wide_resnet50_2(pretrained=False)
            ch_list = [256, 512, 1024]

        self.projector = nn.ModuleList()
        for i in range(len(ch_list)):
            seq = TFA(ch_list[i], ch_list[i])
            self.projector.append(seq)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, src, aug=None):
        self.encoder.eval()
        with torch.no_grad():
            src_list = self.encoder(src)

        src_proj_list = []
        for i in range(len(src_list)):
            src_proj_list.append(self.projector[i](src_list[i]))

        if self.training:
            with torch.no_grad():
                aug_list = self.encoder(aug)
                de_src_list = self.decoder(self.bn(src_list))

            if random.randint(0, 100) < 90:
                for i in range(len(src_list)):
                    aug_list[i] = self.drop(aug_list[i])

            de_aug_list = self.decoder(self.bn(aug_list))
            # with torch.no_grad():

            return src_list, de_aug_list, de_src_list, src_proj_list
        else:
            de_aug_list = self.decoder(self.bn(src_list))
            return src_proj_list, de_aug_list

