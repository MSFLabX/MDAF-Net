# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def img2seq(x):
    [b, c, h, w] = x.shape
    x = x.reshape((b, c, h*w))
    return x

def seq2img(x):
    [b, c, d] = x.shape
    p = int(d ** .5)
    x = x.reshape((b, c, p, p))
    return x

class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()

        # 使用ll替换原有的卷积+池化结构
        self.conv1 = ll(64, 32, 1)  # 假设expand_ratio=1
        self.conv2 = ll(64, 32, 1)
        self.conv1_1 = ll1(32, 64, 2)  # 假设expand_ratio=2
        self.conv2_1 = ll1(32, 64, 2)
        self.conv1_2 = ll1(32, 64, 2)
        self.conv2_2 = ll1(32, 64, 2)
        self.conv1_3 = ll1(32, 64, 2)
        self.conv2_3 = ll1(32, 64, 2)
        self.xishu1 = nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

    def forward(self, x11, x21, x12, x22, x13, x23):
        x11 = self.conv1(x11)
        x21 = self.conv2(x21)
        x12 = self.conv1(x12)
        x22 = self.conv2(x22)
        x13 = self.conv1(x13)
        x23 = self.conv2(x23)

        x1_1 = self.conv1_1(x11)
        x2_1 = self.conv2_1(x21)
        x_add1 = x1_1 * self.xishu1 + x2_1 * self.xishu2

        x1_2 = self.conv1_2(x12)
        x2_2 = self.conv2_2(x22)
        x_add2 = x1_2 * self.xishu1 + x2_2 * self.xishu2

        x1_3 = self.conv1_3(x13)
        x2_3 = self.conv2_3(x23)
        x_add3 = x1_3 * self.xishu1 + x2_3 * self.xishu2

        return x_add1, x_add2, x_add3

class ll(nn.Module):
    def __init__(self, inp, oup, expand_ratio, drop_prob=0.0):
        super(ll, self).__init__()
        self.drop_prob = drop_prob
        self.bottleneckBlock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(inp, oup, 3),
            nn.BatchNorm2d(oup),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.training and torch.rand(1) < self.drop_prob:
            return x
        return self.bottleneckBlock(x)

class ll1(nn.Module):
    def __init__(self, inp, oup, expand_ratio, drop_prob=0.0):
        super(ll1, self).__init__()
        self.drop_prob = drop_prob
        self.bottleneckBlock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(inp, oup, 3),  # 调整输入通道数为32，输出通道数为64
            nn.BatchNorm2d(oup),  # 添加批量归一化
            nn.ReLU(),  # 使用ReLU激活函数
            nn.MaxPool2d(2)  # 添加最大池化
        )
        self.dropout = nn.Dropout2d(p=drop_prob)  # 添加 Dropout 层

    def forward(self, x):
        out = self.bottleneckBlock(x)
        if self.training and torch.rand(1) < self.drop_prob:
            return self.dropout(out)
        return out

class CNN_Encoder2(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Encoder2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

    def forward(self, x11, x21, x12, x22, x13, x23):
        x11 = self.conv1(x11)
        x21 = self.conv2(x21)
        x12 = self.conv1(x12)
        x22 = self.conv2(x22)
        x13 = self.conv1(x13)
        x23 = self.conv2(x23)

        x1_1 = self.conv1_1(x11)
        x2_1 = self.conv2_1(x21)
        x_add1 = x1_1 * self.xishu1 + x2_1 * self.xishu2

        x1_2 = self.conv1_2(x12)
        x2_2 = self.conv2_2(x22)
        x_add2 = x1_2 * self.xishu1 + x2_2 * self.xishu2

        x1_3 = self.conv1_3(x13)
        x2_3 = self.conv2_3(x23)
        x_add3 = x1_3 * self.xishu1 + x2_3 * self.xishu2

        return x_add1, x_add2, x_add3

class CNN_Encoder1(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Encoder1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

    def forward(self, x11, x21, x12, x22, x13, x23):
        x11 = self.conv1(x11)
        x21 = self.conv2(x21)
        x12 = self.conv1(x12)
        x22 = self.conv2(x22)
        x13 = self.conv1(x13)
        x23 = self.conv2(x23)

        x1_1 = self.conv1_1(x11)
        x2_1 = self.conv1_1(x21)

        x1_2 = self.conv1_2(x12)
        x2_2 = self.conv2_2(x22)

        x1_3 = self.conv1_3(x13)
        x2_3 = self.conv2_3(x23)

        return x1_1, x2_1, x1_2, x2_2, x1_3, x2_3

class SharedExchange(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        # 定义一个共享的可学习权重
        self.channel_weight = nn.Parameter(torch.randn(1, channels, 1, 1))  # 通道权重
        self.spatial_weight = nn.Parameter(torch.randn(1, 1, height, width))  # 空间权重

    def forward(self, x1, x2):
        # 通道权重处理
        channel_weight = torch.sigmoid(self.channel_weight)
        out_x1 = channel_weight * x1 + (1 - channel_weight) * x2
        out_x2 = (1 - channel_weight) * x1 + channel_weight * x2

        # 获取输入张量的形状
        _, _, h, w = out_x1.size()

        # 确保空间权重的形状与输入匹配
        if self.spatial_weight.size(2) != h or self.spatial_weight.size(3) != w:
            spatial_weight = F.interpolate(self.spatial_weight, size=(h, w), mode='bilinear', align_corners=False)
        else:
            spatial_weight = self.spatial_weight

        # 使用 sigmoid 将空间权重限制在 [0, 1]
        spatial_weight = torch.sigmoid(spatial_weight)

        # 空间权重处理
        out_x1 = spatial_weight * out_x1 + (1 - spatial_weight) * out_x2
        out_x2 = (1 - spatial_weight) * out_x1 + spatial_weight * out_x2

        return out_x1, out_x2


class SC(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.shared_exchange = SharedExchange(channels, height, width)

    def forward(self, image1, image2):
        out_x1, out_x2 = self.shared_exchange(image1, image2)

        # 使用双三次插值进行上采样
        out_x1 = F.interpolate(out_x1, scale_factor=2, mode='bicubic', align_corners=False)
        out_x2 = F.interpolate(out_x2, scale_factor=2, mode='bicubic', align_corners=False)

        out_x1, out_x2 = self.shared_exchange(out_x1, out_x2)

        return out_x1, out_x2


class CNN_Decoder(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Decoder, self).__init__()

        self.dconv1 = nn.Sequential(
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv5 = nn.Sequential(
            nn.Upsample(scale_factor=3),  # add Upsample
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv6 = nn.Sequential(
            nn.Upsample(scale_factor=3),  # add Upsample
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )

    def forward(self, x_con1):
        x1 = self.dconv1(x_con1)
        x2 = self.dconv2(x_con1)

        x3 = self.dconv3(x_con1)
        x4 = self.dconv4(x_con1)

        x5 = self.dconv5(x_con1)
        x6 = self.dconv6(x_con1)
        return x1, x2, x3, x4, x5, x6

class CNN_Classifier(nn.Module):
    def __init__(self, Classes):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x_out = F.softmax(x, dim=1)
        return x_out


class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, msf, panf):
        _, _, H, W = msf.shape

        # Preprocess inputs and apply Fourier transform
        msF = torch.fft.rfft2(self.pre1(msf) + 1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf) + 1e-8, norm='backward')

        # Amplitude and phase extraction with normalization
        msF_amp = torch.abs(msF)
        msF_amp = msF_amp / (msF_amp.max() + 1e-6)  # Amplitude normalization
        msF_pha = torch.angle(msF)

        panF_amp = torch.abs(panF)
        panF_amp = panF_amp / (panF_amp.max() + 1e-6)  # Amplitude normalization
        panF_pha = torch.angle(panF)

        # Fuse amplitude and phase information
        amp_fuse = self.amp_fuse(torch.cat([msF_amp, panF_amp], dim=1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha, panF_pha], dim=1))

        # Combine amplitude and phase into a complex tensor
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        # Apply inverse Fourier transform and normalize output
        out = torch.fft.irfft2(out, s=(H, W), norm='backward')
        out = torch.abs(out)  # Ensure non-negative output
        out = out / (out.max() + 1e-6)  # Final normalization
        return self.post(out)



class MDAF(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes, encoder_embed_dim, decoder_embed_dim):
        super().__init__()
        self.cnn_encoder = CNN_Encoder()
        self.cnn_encoder2 = CNN_Encoder2(l1, l2)
        self.cnn_encoder1 = CNN_Encoder1(l1, l2)
        self.sc1 = SC(64,4,4)
        self.sc2 = SC(64,8,8)
        self.sc3 = SC(64,12,12)
        self.cnn_decoder = CNN_Decoder(l1, l2)
        self.cnn_classifier = CNN_Classifier(num_classes)
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.loss_fun2 = nn.MSELoss()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim

        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, encoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, decoder_embed_dim))
        self.encoder_embedding1 = nn.Linear(((patch_size // 2) * 1) ** 2, self.patch_size ** 2)
        self.encoder_embedding2 = nn.Linear(((patch_size // 2) * 2) ** 2, self.patch_size ** 2)
        self.encoder_embedding3 = nn.Linear(((patch_size // 2) * 3) ** 2, self.patch_size ** 2)
        self.decoder_embedding = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))

        self.decoder_pred1 = nn.Linear(decoder_embed_dim, 64, bias=True)  # decoder to patch
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, num_classes)
        )
        self.fre_process = Freprocess(64)
        # self.fuse1 = FeatureProcess(64)
        # self.fuse2 = Net(64)

    def encoder(self, x11, x21, x12, x22, x13, x23):
        x1_1, x2_1, x1_2, x2_2, x1_3, x2_3 = self.cnn_encoder1(x11, x21, x12, x22, x13, x23)  # 通过CNN编码器处理输入特征图

        f11, f21 = self.sc1(x1_1, x2_1)  # 通过CNN编码器处理输入特征图
        f12, f22 = self.sc2(x1_2, x2_2)  # 通过CNN编码器处理输入特征图
        f13, f23 = self.sc3(x1_3, x2_3)  # 通过CNN编码器处理输入特征图

        x_fuse1, x_fuse2, x_fuse3 = self.cnn_encoder(f11, f21, f12, f22, f13, f23)

        ff1 = self.fre_process(x1_1, x2_1)
        ff2 = self.fre_process(x1_2, x2_2)
        ff3 = self.fre_process(x1_3, x2_3)

        x_fuse1 = x_fuse1 + ff1
        x_fuse2 = x_fuse2 + ff2
        x_fuse3 = x_fuse3 + ff3

        x_flat1 = x_fuse1.flatten(2)
        x_flat2 = x_fuse2.flatten(2)
        x_flat3 = x_fuse3.flatten(2)

        x_1 = self.encoder_embedding1(x_flat1)
        x_2 = self.encoder_embedding2(x_flat2)
        x_3 = self.encoder_embedding3(x_flat3)

        x_cnn = x_1 + x_2 + x_3

        # x_fuse2 = self.conv1(x_fuse2)
        # x_fuse3 = self.conv2(x_fuse3)
        #
        # x_cnn = self.fuse1(x_fuse1, x_fuse2, x_fuse3)
        return x_cnn

    def classifier(self, x_cnn):
        x_cls2 = self.cnn_classifier(seq2img(x_cnn))
        return x_cls2

    def forward(self, img11, img21, img12, img22, img13, img23):
        x_cnn = self.encoder(img11, img21, img12, img22, img13, img23)
        x_cls = self.classifier(x_cnn)
        return x_cls

