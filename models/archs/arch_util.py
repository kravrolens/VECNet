import ipdb
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResidualBlock_BN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_BN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv1_bn(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        return identity + out


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample='pool', bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample: 'pool' or 'bilinear' or False
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            if sub_sample == 'pool':
                max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            elif sub_sample == 'bilinear':
                max_pool_layer = nn.UpsamplingBilinear2d([16, 16])
            else:
                raise NotImplementedError(f'[ ERR ] Unknown down sample method: {sub_sample}')
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample='pool', bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, )


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, input):
        '''
        coeff shape: [bs, 12, h, w]
        input shape: [bs, 3, h, w]
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        R = torch.sum(input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class ModulatedDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super(ModulatedDeformConv, self).__init__()

        self.dcnpack = DeformConv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups=groups)

        self.conv_offset_mask = nn.Conv2d(in_channels, groups * 3 * kernel_size * kernel_size,
                                          kernel_size, stride, padding, bias=True)

        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x, y):
        out = self.conv_offset_mask(y)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.dcnpack(x, offset, mask)


class PhaAL(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(PhaAL, self).__init__()

        # TODO: alignment
        # CVPR24初稿的版本
        # self.offset_conv1 = nn.Conv2d(in_nc * 2, in_nc, 3, 1, 1, bias=True)
        # self.offset_conv2 = nn.Conv2d(in_nc, in_nc, 3, 1, 1, bias=True)

        # CVPR24修改的版本
        self.offset_conv1 = nn.Conv2d(in_nc * 2, in_nc, 1, 1, 0, bias=True)
        self.offset_conv2 = nn.Conv2d(in_nc, in_nc, 1, 1, 0, bias=True)

        # self.dcnpack = DCN(in_nc, in_nc, 3, stride=1, padding=1, dilation=1)
        self.dcnpack = ModulatedDeformConv(in_nc, in_nc, 3, stride=1, padding=1, dilation=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.cat = nn.Conv2d(2 * in_nc, out_nc, 1, 1, 0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.process = nn.Sequential(nn.Conv2d(in_nc * 2, in_nc // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(in_nc // 2, in_nc * 2, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())

    def forward(self, x, avg_amp, ref):
        # x_amp_freq = torch.fft.rfft2(x_amp, norm='backward')
        x_freq = torch.fft.rfft2(x, norm='backward')
        ref_freq = torch.fft.rfft2(ref, norm='backward')

        # x_amp_freq_amp = torch.abs(avg_amp)  # Real part
        # x_freq_pha = torch.angle(x_freq)
        x_freq_pha = torch.angle(x_freq)  # Imaginary part
        ref_freq_pha = torch.angle(ref_freq)  # Imaginary part

        offset = torch.cat([x_freq_pha, ref_freq_pha], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))

        deform_feat = self.lrelu(self.dcnpack(x_freq_pha, offset))   # !!!! problem here

        real = avg_amp * torch.cos(deform_feat)
        imag = avg_amp * torch.sin(deform_feat)
        x_recom = torch.complex(real, imag) + 1e-8
        x_recom = torch.fft.irfft2(x_recom) + 1e-8
        x_recom = torch.abs(x_recom) + 1e-8
        xcat = torch.cat([x_recom, x], 1)
        xcat = self.process(self.contrast(xcat) + self.avgpool(xcat)) * xcat
        x_out = self.cat(xcat)

        return x_out


class FourAL(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(FourAL, self).__init__()

        self.offset_conv1 = nn.Conv2d(in_nc * 2, in_nc, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(in_nc, in_nc, 3, 1, 1, bias=True)

        self.dcnpack = ModulatedDeformConv(in_nc, in_nc, 3, stride=1, padding=1, dilation=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv1x1 = nn.Conv2d(in_nc, out_nc, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(in_nc * 2, in_nc, 3, 1, 1)

    def forward(self, x, x_ref, amp_avg, new_inp):
        # x: the frame to be aligned
        # x_ref: the center frame
        # amp_avg: the average amplitude of the frames

        x_freq = torch.fft.rfft2(x)
        ref_freq = torch.fft.rfft2(x_ref)

        # use amp_avg
        x_freq_phase = torch.angle(x_freq)
        ref_freq_phase = torch.angle(ref_freq)
        x_freq_new = amp_avg * torch.exp(1j * x_freq_phase)
        ref_freq_new = amp_avg * torch.exp(1j * ref_freq_phase)
        x_new = torch.fft.irfft2(x_freq_new, norm='backward').real
        ref_new = torch.fft.irfft2(ref_freq_new, norm='backward').real

        offset = torch.cat([x_new, ref_new], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))

        deform_x = self.lrelu(self.dcnpack(x, offset))
        deform_x = self.lrelu(self.conv1x1(deform_x))

        x_total = torch.cat([deform_x, new_inp], dim=1)
        x_total = self.conv3x3(x_total)

        return x_total
