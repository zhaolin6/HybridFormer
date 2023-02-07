import torch
import math
from torch import nn
from torch.nn import functional as F

class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding,stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x = x.squeeze(-1)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = (self.gamma*out + x).unsqueeze(-1)
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, channle = x.size()
        #print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) #形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        out = self.gamma*out + x  #C*H*W
        return out


class DBDA_network_MISH(nn.Module):
    def __init__(self, band, classes):
        super(DBDA_network_MISH, self).__init__()
        # spectral branch
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)) # kernel size随数据变化


        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    mish()
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    mish()
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))


        # self.conv25 = nn.Sequential(
        #                         nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
        #                         kernel_size=(3, 3, 2), stride=(1, 1, 1)),
        #                         nn.Sigmoid()
        # )

        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish(),
                                    nn.Dropout(p=0.5)
        )


        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                #nn.Dropout(p=0.5),
                                nn.Linear(120, classes) # ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

    def forward(self, X):
        # spectral
        if len(X.shape) == 4:
            X = X.unsqueeze(dim=1)
        X = X.permute(0, 1, 3, 4, 2)
        x11 = self.conv11(X)

        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)

        x13 = torch.cat((x11, x12), dim=1)

        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)

        # spatial
        #print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)

        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.batch_norm_spatial(x2)
        x2= self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        output = self.full_connection(x_pre)

        return output


class SSRN_network(nn.Module):
    def __init__(self, band, classes):
        super(SSRN_network, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.res_net1 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(24, 24, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(24, 24, (3, 3, 1), (1, 1, 0))

        kernel_3d = math.ceil((band - 6) / 2)

        self.conv2 = nn.Conv3d(in_channels=24, out_channels=128, padding=(0, 0, 0),
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0),
                               kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(96, classes)  # ,
            # nn.Softmax()
        )

    def forward(self, X):
        if len(X.shape) == 4:
            X = X.unsqueeze(dim=1)
        X = X.permute(0, 1, 3, 4, 2)
        x1 = self.batch_norm1(self.conv1(X))
        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        # print(x10.shape)
        return self.full_connection(x4)

class FDSSC_network(nn.Module):
    def __init__(self, band, classes):
        super(FDSSC_network, self).__init__()

        # spectral branch
        self.name = 'FDSSC'
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm1 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    nn.PReLU()
        )
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    nn.PReLU()
        )
        self.conv3 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    nn.PReLU()
        )
        self.conv4 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm4 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.ceil((band - 6) / 2)
        # print(kernel_3d)
        self.conv5 = nn.Conv3d(in_channels=60, out_channels=200, padding=(0, 0, 0),
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.batch_norm5 = nn.Sequential(
                                    nn.BatchNorm3d(1, eps=0.001, momentum=0.1, affine=True),
                                    nn.PReLU()
        )
        self.conv6 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0), kernel_size=(1, 1, 200), stride=(1, 1, 1))

        self.batch_norm6 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.PReLU()
        )
        self.conv7 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm7 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv8 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm8 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv9 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm9 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(60, classes)
                                # nn.Softmax()
        )


    def forward(self, X):
        # spectral
        if len(X.shape) == 4:
            X = X.unsqueeze(dim=1)
        X = X.permute((0, 1, 3, 4, 2))
        x1 = self.conv1(X)
        x2 = self.batch_norm1(x1)
        x2 = self.conv2(x2)

        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.batch_norm2(x3)
        x3 = self.conv3(x3)
        #print('x13', x13.shape)

        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.batch_norm3(x4)
        x4 = self.conv4(x4)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        # print('x15', x15.shape)

        # print(x5.shape)
        x6 = self.batch_norm4(x5)
        x6 = self.conv5(x6)
        #print('x16', x16.shape)  # 7*7*97, 60

        #print('x16', x16.shape)
        # 光谱注意力通道
        x6 = x6.permute((0, 4, 2, 3, 1))
        # print(x6.shape)

        x7 = self.batch_norm5(x6)
        x7 = self.conv6(x7)

        x8 = self.batch_norm6(x7)
        x8 = self.conv7(x8)

        x9 = torch.cat((x7, x8), dim=1)
        x9 = self.batch_norm7(x9)
        x9 = self.conv8(x9)

        x10 = torch.cat((x7, x8, x9), dim=1)
        x10 = self.batch_norm8(x10)
        x10 = self.conv9(x10)

        x10 = torch.cat((x7, x8, x9, x10), dim=1)
        x10 = self.batch_norm9(x10)
        x10 = self.global_pooling(x10)
        x10 = x10.view(x10.size(0), -1)

        output = self.full_connection(x10)
        # output = self.fc(x_pre)
        return output