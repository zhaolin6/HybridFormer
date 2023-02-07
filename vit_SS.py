import random
import numpy as np
import torch
import cv2
import math
import torch.nn.functional as F
from torch import nn
from skimage import transform
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class OurFE(nn.Module):
    def __init__(self, channel, dim):
        super(OurFE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(3 * channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out = self.out_conv(torch.cat((out1, out2, out3), dim=1))
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            DEPTHWISECONV(dim, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        b, d, c = x.shape
        w = int(math.sqrt(d))
        x1 = rearrange(x, 'b (w h) c -> b c w h', w=w, h=w)
        x1 = self.net(x1)
        x1 = rearrange(x1, 'b c w h -> b (w h) c')
        x = x + x1
        return x


class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, padding=0, stride=1, is_fe=False):
        super(DEPTHWISECONV, self).__init__()
        self.is_fe = is_fe
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        if self.is_fe:
            return out
        out = self.point_conv(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0., num_patches=10):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.spatial_norm = nn.BatchNorm2d(heads)
        self.spatial_conv = nn.Conv2d(heads, heads, kernel_size=3, padding=1)

        self.spectral_norm = nn.BatchNorm2d(1)
        self.spectral_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.to_qkv_spec = nn.Linear(num_patches, num_patches*3, bias=False)
        self.attend_spec = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.spatial_conv(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        output = self.to_out(out)

        x = x.transpose(-2, -1)
        qkv_spec = self.to_qkv_spec(x).chunk(3, dim=-1)
        q_spec, k_spec, v_spec = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=1), qkv_spec)
        dots_spec = torch.matmul(q_spec, k_spec.transpose(-1, -2)) * self.scale
        attn = self.attend_spec(dots_spec)  # .squeeze(dim=1)
        attn = self.spectral_conv(attn).squeeze(dim=1)

        return torch.matmul(output, attn)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., num_patches=25):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.index = 0
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim)),
            ]))

    def forward(self, x):
        output = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            output.append(x)

        return x, output


class SubNet(nn.Module):
    def __init__(self, patch_size, num_patches, dim, emb_dropout, depth, heads, dim_head, mlp_dim, dropout):
        super(SubNet, self).__init__()
        self.to_patch_embedding = nn.Sequential(
            DEPTHWISECONV(in_ch=dim, out_ch=dim, kernel_size = patch_size, stride = patch_size, padding=0, is_fe=True),
            Rearrange('b c w h -> b (h w) c '),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches+1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, dropout=dropout, num_patches=num_patches)


def get_num_patches(ps, ks):
    return int((ps - ks)/ks)+1


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super(ViT, self).__init__()
        self.ournet = OurFE(channels, dim)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=1)
        self.net = nn.Sequential()
        self.mlp_head = nn.ModuleList()
        for ps in patch_size:
            num_patches = get_num_patches(image_size, ps) ** 2
            patch_dim = dim * num_patches
            sub_net = SubNet(ps, num_patches, dim, emb_dropout, depth, heads, dim_head, mlp_dim, dropout)
            self.net.append(sub_net)
            self.mlp_head.append(nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, num_classes)
            ))

        self.weight = torch.ones(len(patch_size))

    def forward(self, img):
        if len(img.shape) == 5: img = img.squeeze()
        img = self.ournet(img)
        img = self.pool(img)
        img = self.conv4(img)

        all_branch = []
        for sub_branch in self.net:
            spatial = sub_branch.to_patch_embedding(img)
            b, n, c = spatial.shape
            spatial = spatial + sub_branch.pos_embedding[:, :n]
            spatial = sub_branch.dropout(spatial)
            _, outputs = sub_branch.transformer(spatial)
            res = outputs[-1]
            all_branch.append(res)

        self.weight = F.softmax(self.weight, 0)
        res = 0
        for i, mlp_head in enumerate(self.mlp_head):
            out1 = all_branch[i].flatten(start_dim=1)
            cls1 = mlp_head(out1)
            res = res + cls1 * self.weight[i]
        return res
