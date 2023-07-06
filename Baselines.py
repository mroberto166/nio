# Credit to Deng et al. https://arxiv.org/pdf/2111.02926.pdf
# Code from https://openfwi-lanl.github.io/index.html

from collections import OrderedDict
from math import ceil

import torch.nn as nn
import torch.nn.functional as F

from debug_tools import *

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}


# Replace the key names in the checkpoint in which legacy network building blocks are used
def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
             .replace('Conv2DwithBN_Tanh', 'layers')
             .replace('Deconv2DwithBN', 'layers')
             .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 1000, 70 -> 70, 70
class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)
        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        x = self.convblock7_1(x)  # (None, 256, 8, 9) 7
        x = self.convblock7_2(x)  # (None, 256, 8, 9)
        x = self.convblock8(x)  # (None, 512, 1, 1)

        # Decoder Part
        x = self.deconv1_1(x)  # (None, 512, 5, 5)
        x = self.deconv1_2(x)  # (None, 512, 5, 5)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20) 32, 28
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40) 64, 56
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80) 128, 112
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x.squeeze(1)

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class EncoderInversionNet(nn.Module):
    def __init__(self, n_out, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(EncoderInversionNet, self).__init__()
        # self.n_out = n_out
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        self.linear = nn.Linear(512, n_out)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)
        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        x = self.convblock7_1(x)  # (None, 256, 8, 9) 7
        x = self.convblock7_2(x)  # (None, 256, 8, 9)
        x = self.convblock8(x)  # (None, 512, 1, 1)
        x = nn.Flatten()(x)
        # if self.n_out != 512:
        #    x = self.linear(x)
        x = self.linear(x)
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class InversionNetHelm(nn.Module):
    def __init__(self, start, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, print_bool=False):
        super(InversionNetHelm, self).__init__()
        dim1, dim2, dim3, dim4 = [start * 2 ** i for i in range(4)]
        self.print_bool = print_bool
        self.convblock1 = ConvBlock(4, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim2, stride=2)
        self.convblock4_2 = ConvBlock(dim2, dim2)
        self.convblock5_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=(5, 5), padding=0)

        self.deconv1_1 = DeconvBlock(dim4, dim4, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim4, dim4)
        self.deconv2_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim3, dim3)
        self.deconv3_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim2, dim2)
        self.deconv4_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim1, dim1)
        self.deconv5_1 = DeconvBlock(dim1, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x, grid):
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        if self.print_bool: print(x.shape)
        x = self.convblock1(x)  # (None, 32, 49, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        # x = self.convblock3_1(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        # x = self.convblock3_2(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        # x = self.convblock6_2(x)  # (None, 256, 16, 18)
        # if self.print_bool: print(x.shape)
        # x = self.convblock7_1(x)  # (None, 256, 8, 9) 7
        # if self.print_bool: print(x.shape)
        # x = self.convblock7_2(x)  # (None, 256, 8, 9)
        # if self.print_bool: print(x.shape)
        # x = self.convblock8(x)  # (None, 512, 1, 1)
        # if self.print_bool: print(x.shape)

        # Decoder Part
        if self.print_bool: print(x.shape)
        x = self.deconv1_1(x)  # (None, 512, 5, 5)
        if self.print_bool: print(x.shape)
        x = self.deconv1_2(x)  # (None, 512, 5, 5)
        if self.print_bool: print(x.shape)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        if self.print_bool: print(x.shape)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        if self.print_bool: print(x.shape)
        x = self.deconv3_1(x)  # (None, 128, 20, 20) 32, 28
        if self.print_bool: print(x.shape)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        if self.print_bool: print(x.shape)
        x = self.deconv4_1(x)  # (None, 64, 40, 40) 64, 56
        if self.print_bool: print(x.shape)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        if self.print_bool: print(x.shape)
        x = self.deconv5_1(x)  # (None, 32, 80, 80) 128, 112
        if self.print_bool: print(x.shape)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        if self.print_bool: print(x.shape)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        if self.print_bool: print(x.shape)
        x = self.deconv6(x)  # (None, 1, 70, 70)
        if self.print_bool: print(x.shape)
        x = x.squeeze(1)
        if self.print_bool: print(x.shape)
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class EncoderHelm(nn.Module):
    def __init__(self, n_out, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, print_bool=False, **kwargs):
        super(EncoderHelm, self).__init__()
        # self.n_out = n_out
        self.convblock1 = ConvBlock(4, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        # self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim2, stride=2)
        self.convblock4_2 = ConvBlock(dim2, dim2)
        self.convblock5_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=(5, 5), padding=0)
        self.linear = nn.Linear(512, n_out)
        self.print_bool = print_bool

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        if self.print_bool: print(x.shape)
        x = self.convblock1(x)  # (None, 32, 49, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        # x = self.convblock3_1(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        # x = self.convblock3_2(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        # x = self.convblock6_2(x)  # (None, 256, 16, 18)
        # if self.print_bool: print(x.shape)
        # x = self.convblock7_1(x)  # (None, 256, 8, 9) 7
        # if self.print_bool: print(x.shape)
        # x = self.convblock7_2(x)  # (None, 256, 8, 9)
        # if self.print_bool: print(x.shape)
        # x = self.convblock8(x)  # (None, 512, 1, 1)
        # if self.print_bool: print(x.shape)
        x = nn.Flatten()(x)
        if self.print_bool: print(x.shape)
        # if self.n_out != 512:
        #    x = self.linear(x)
        x = self.linear(x)
        if self.print_bool: print(x.shape)
        if self.print_bool: quit()
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class EncoderHelm2(nn.Module):
    def __init__(self, n_out, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, print_bool=False, **kwargs):
        super(EncoderHelm2, self).__init__()
        # self.n_out = n_out
        self.convblock1 = ConvBlock(1, dim1, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(1, 3), padding=(0, 1))
        self.convblock3_1 = ConvBlock(dim2, dim3, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.convblock3_2 = ConvBlock(dim3, dim3, kernel_size=(1, 3), padding=(0, 1))
        self.convblock4_1 = ConvBlock(dim3, dim4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.convblock4_2 = ConvBlock(dim4, dim4, kernel_size=(1, 3), padding=(0, 1))
        # self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock7_1 = ConvBlock(dim4, dim5, kernel_size=(4, 5), padding=0)
        self.linear = nn.Linear(512, n_out)
        self.print_bool = print_bool

    def forward(self, x):
        batch_size = x.shape[0]
        size_fun = x.shape[1]
        x = x.view(batch_size * size_fun, x.shape[2], x.shape[3], x.shape[4])
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        if self.print_bool: print(x.shape)
        x = self.convblock1(x)  # (None, 32, 49, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock3_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock3_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        # x = self.convblock3_1(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        # x = self.convblock3_2(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock7_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        # x = self.convblock6_2(x)  # (None, 256, 16, 18)
        # if self.print_bool: print(x.shape)
        # x = self.convblock7_1(x)  # (None, 256, 8, 9) 7
        # if self.print_bool: print(x.shape)
        # x = self.convblock7_2(x)  # (None, 256, 8, 9)
        # if self.print_bool: print(x.shape)
        # x = self.convblock8(x)  # (None, 512, 1, 1)
        # if self.print_bool: print(x.shape)
        x = nn.Flatten()(x)

        if self.print_bool: print(x.shape)
        x = x.view(batch_size, size_fun, x.shape[1])
        if self.print_bool: print(x.shape)
        # if self.n_out != 512:
        #    x = self.linear(x)
        x = self.linear(x)
        if self.print_bool: print(x.shape)

        if self.print_bool: print(x.shape)
        if self.print_bool: quit()
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class InversionNetRad(nn.Module):
    def __init__(self, start, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, print_bool=False):
        super(InversionNetRad, self).__init__()
        dim1, dim2, dim3, dim4, dim5 = [start * 2 ** i for i in range(5)]
        self.print_bool = print_bool
        self.convblock1 = ConvBlock(1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim1, kernel_size=5, stride=2, padding=2)
        self.convblock2_2 = ConvBlock(dim1, dim2)
        self.convblock4_1 = ConvBlock(dim2, dim2, stride=2)
        self.convblock4_2 = ConvBlock(dim2, dim3)
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim4)
        self.convblock6_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim5)
        self.convblock7_1 = ConvBlock(dim5, dim5, stride=2)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=(5, 1))
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x, grid):
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        if self.print_bool: print(x.shape)
        x = self.convblock1(x)  # (None, 32, 49, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        # x = self.convblock3_1(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        # x = self.convblock3_2(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = self.convblock7_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)

        # Decoder Part
        if self.print_bool: print(x.shape)
        x = self.deconv1_1(x)  # (None, 512, 5, 5)
        if self.print_bool: print(x.shape)
        x = self.deconv1_2(x)  # (None, 512, 5, 5)
        if self.print_bool: print(x.shape)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        if self.print_bool: print(x.shape)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        if self.print_bool: print(x.shape)
        x = self.deconv3_1(x)  # (None, 128, 20, 20) 32, 28
        if self.print_bool: print(x.shape)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        if self.print_bool: print(x.shape)
        x = self.deconv4_1(x)  # (None, 64, 40, 40) 64, 56
        if self.print_bool: print(x.shape)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        if self.print_bool: print(x.shape)
        x = self.deconv5_1(x)  # (None, 32, 80, 80) 128, 112
        if self.print_bool: print(x.shape)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        if self.print_bool: print(x.shape)
        x = F.pad(x, [0, 0, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        if self.print_bool: print(x.shape)
        x = self.deconv6(x)  # (None, 1, 70, 70)
        if self.print_bool: print(x.shape)
        x = x.squeeze(1)
        x = x.squeeze(2)
        if self.print_bool:
            print(x.shape)
            quit()
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class EncoderRad(nn.Module):
    def __init__(self, n_out, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, print_bool=False, **kwargs):
        super(EncoderRad, self).__init__()
        # self.n_out = n_out
        self.print_bool = print_bool
        self.convblock1 = ConvBlock(1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim1, kernel_size=5, stride=2, padding=2)
        self.convblock2_2 = ConvBlock(dim1, dim2)
        self.convblock4_1 = ConvBlock(dim2, dim2, stride=2)
        self.convblock4_2 = ConvBlock(dim2, dim3)
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim4)
        self.convblock6_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim5)
        self.convblock7_1 = ConvBlock(dim5, dim5, stride=2)
        self.linear = nn.Linear(dim5, n_out)
        self.print_bool = print_bool

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        if self.print_bool: print(x.shape)
        x = self.convblock1(x)  # (None, 32, 49, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = self.convblock7_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = nn.Flatten()(x)
        if self.print_bool: print(x.shape)
        # if self.n_out != 512:
        #    x = self.linear(x)
        x = self.linear(x)
        if self.print_bool: print(x.shape)
        if self.print_bool: quit()
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class EncoderRad2(nn.Module):
    def __init__(self, n_out, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, print_bool=False, **kwargs):
        super(EncoderRad2, self).__init__()
        # self.n_out = n_out
        self.print_bool = print_bool
        self.convblock1 = ConvBlock(1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim1, kernel_size=5, stride=2, padding=2)
        self.convblock2_2 = ConvBlock(dim1, dim2)
        self.convblock4_1 = ConvBlock(dim2, dim2, stride=2)
        self.convblock4_2 = ConvBlock(dim2, dim3)
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim4)
        self.convblock6_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim5)
        self.convblock7_1 = ConvBlock(dim5, dim5, stride=2)
        self.linear = nn.Linear(dim5, n_out)
        self.print_bool = print_bool

    def forward(self, x):
        if self.print_bool: print(x.shape)
        x = x.unsqueeze(3)
        if self.print_bool: print(x.shape)
        # print(x.shape)
        batch_size = x.shape[0]
        size_fun = x.shape[1]
        '''import matplotlib.pyplot as plt
        import numpy as np
        plt.figure()
        for i in [0, 5, 10]:
            plt.plot(np.arange(32), x[0, :, :,0, i].detach().squeeze(0))

        plt.figure()
        for i in [0, 5, 10]:
            plt.plot(np.arange(32), x[0, i, :,0, :].detach().squeeze(0))

        plt.show()
        quit()'''
        if self.print_bool: print(x.shape)
        x = x.reshape(batch_size * size_fun, x.shape[2], x.shape[3], x.shape[4])
        if self.print_bool: print(x.shape)
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        if self.print_bool: print(x.shape)
        x = self.convblock1(x)  # (None, 32, 49, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = self.convblock7_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = nn.Flatten()(x)
        if self.print_bool: print(x.shape)
        x = x.reshape(batch_size, size_fun, x.shape[1])
        # if self.n_out != 512:
        #    x = self.linear(x)
        x = self.linear(x)
        if self.print_bool: print(x.shape)
        if self.print_bool: quit()
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class InversionNetEIT(nn.Module):
    def __init__(self, start, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, print_bool=False):
        super(InversionNetEIT, self).__init__()
        dim1, dim2, dim3, dim4, dim5 = [start * 2 ** i for i in range(5)]
        self.print_bool = print_bool
        self.convblock1 = ConvBlock(1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim1, kernel_size=5, stride=2, padding=2)
        self.convblock2_2 = ConvBlock(dim1, dim2)
        self.convblock4_1 = ConvBlock(dim2, dim2, stride=2)
        self.convblock4_2 = ConvBlock(dim2, dim3)
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim4)
        self.convblock6_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim5)
        self.convblock7_1 = ConvBlock(dim5, dim5, stride=2)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=(5, 5))
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x, grid):
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        if self.print_bool: print(x.shape)
        x = self.convblock1(x)  # (None, 32, 49, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        # x = self.convblock3_1(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        # x = self.convblock3_2(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        if self.print_bool: print(x.shape)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        x = self.convblock7_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)

        # Decoder Part
        if self.print_bool: print(x.shape)
        x = self.deconv1_1(x)  # (None, 512, 5, 5)
        if self.print_bool: print(x.shape)
        x = self.deconv1_2(x)  # (None, 512, 5, 5)
        if self.print_bool: print(x.shape)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        if self.print_bool: print(x.shape)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        if self.print_bool: print(x.shape)
        x = self.deconv3_1(x)  # (None, 128, 20, 20) 32, 28
        if self.print_bool: print(x.shape)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        if self.print_bool: print(x.shape)
        x = self.deconv4_1(x)  # (None, 64, 40, 40) 64, 56
        if self.print_bool: print(x.shape)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        if self.print_bool: print(x.shape)
        x = self.deconv5_1(x)  # (None, 32, 80, 80) 128, 112
        if self.print_bool: print(x.shape)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        if self.print_bool: print(x.shape)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        if self.print_bool: print(x.shape)
        x = self.deconv6(x)  # (None, 1, 70, 70)
        if self.print_bool: print(x.shape)
        x = x.squeeze(1)
        x = x.squeeze(2)
        if self.print_bool:
            print(x.shape)
            quit()
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class EncoderInversionNet2(nn.Module):
    def __init__(self, n_out, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(EncoderInversionNet2, self).__init__()
        # self.n_out = n_out
        self.convblock1 = ConvBlock(1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        self.linear = nn.Linear(512, n_out)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.unsqueeze(2)
        batch_size = x.shape[0]
        size_fun = x.shape[1]
        x = x.reshape(batch_size * size_fun, x.shape[2], x.shape[3], x.shape[4])
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)
        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        x = self.convblock7_1(x)  # (None, 256, 8, 9) 7
        x = self.convblock7_2(x)  # (None, 256, 8, 9)
        x = self.convblock8(x)  # (None, 512, 1, 1)
        x = nn.Flatten()(x)
        x = x.reshape(batch_size, size_fun, x.shape[1])
        # if self.n_out != 512:
        #    x = self.linear(x)
        x = self.linear(x)
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
