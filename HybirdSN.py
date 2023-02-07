import torch
import torch.nn as nn


class ChannelAttention(nn.Module):  # 通道注意力机制
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):  # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class HybirdSN(nn.Module):
    def __init__(self, num_classes, channels,  self_attention=True):
        super(HybirdSN, self).__init__()
        # 默认不使用注意力机制

        self.self_attention = False

        # 3D卷积块
        self.block_1_3D = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(7, 3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=(5, 3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True)
        )
        if self_attention:
            self.channel_attention_1 = ChannelAttention(2912)  # 此处的是，
            self.spatial_attention_1 = SpatialAttention(kernel_size=7)
        # 2D卷积块
        t_channels = 32 * (channels - 12)
        self.block_2_2D = nn.Sequential(
            nn.Conv2d(
                in_channels=(t_channels),
                out_channels=64,
                kernel_size=(3, 3)
            ),
            nn.ReLU(inplace=True)
        )
        if self_attention:
            self.channel_attention_2 = ChannelAttention(64)
            self.spatial_attention_2 = SpatialAttention(kernel_size=7)

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=3136,  # 18496,10816
                out_features=256
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
        )
        # self.sigmoid = nn.Sigmoid(dim=-1)
        self.soft = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(dim=1)

        y = self.block_1_3D(x)
        y = y.view(-1, y.shape[1] * y.shape[2], y.shape[3], y.shape[4])
        # print(y.shape)
        if self.self_attention:
            y = self.channel_attention_1(y) * y
            y = self.spatial_attention_1(y) * y
        y = self.block_2_2D(y)
        # print(y.shape)
        if self.self_attention:
            y = self.channel_attention_2(y) * y
            y = self.spatial_attention_2(y) * y
        # y = torch.flatten(y.detach())

        y = y.view(y.size(0), -1)
        # print(y.shape)
        y = self.classifier(y)
        # y = self.soft(y)

        return y
        # return self.sigmoid(y)
        # 全连接层
