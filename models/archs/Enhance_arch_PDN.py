import yaml

from models.LAN.NAFNet import *
from models.LAN.lldm.vqvae import VQGANLight
from models.LAN.lldm.unet import UNetLight
from models.LAN.lldm import LLDM
from utils.util import load_checkpoint
from models.archs.arch_util import NONLocalBlock2D, ApplyCoeffs, FourAL
from models.archs.Enhance_arch_ENC import AttING


class BaselinePDN(nn.Module):

    def __init__(self, lldm_config, vae_config, unet_config, lldm_ckpt, down_blocks, frozen_lldm=True, nframes=1,
                 img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dw_expand=1,
                 ffn_expand=2):
        super().__init__()
        latent_dim, self.illumination_module = self.get_illumination_module(lldm_config, vae_config, unet_config,
                                                                            lldm_ckpt)
        if frozen_lldm:
            for param in self.illumination_module.parameters():
                param.requires_grad = False
        else:
            for param in self.illumination_module.parameters():
                param.requires_grad = True

        self.nframes = nframes
        self.center = nframes // 2

        self.frames_Projection = nn.ModuleList(
            [framesBlock(channels=[32, 16, 16, 3], block=ConvBlock) for _ in range(nframes)])

        self.fal = nn.ModuleList([FourAL(img_channel, img_channel) for _ in range(nframes)])

        self.amp_map = nn.Conv2d(in_channels=nframes * img_channel, out_channels=img_channel, kernel_size=1)
        self.pha_maps = nn.ModuleList(
            [nn.Conv2d(in_channels=img_channel, out_channels=img_channel, kernel_size=1) for _ in range(nframes)])

        # TODO: change here
        # self.attin = AttIN(in_channels=width, channels=width)
        self.attin = AttING(in_channels=width, channels=width)

        self.intro = nn.Conv2d(in_channels=nframes * img_channel, out_channels=width, kernel_size=3, padding=1,
                               stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)
        self.end_illu = nn.Conv2d(in_channels=width, out_channels=img_channel * 4, kernel_size=3, padding=1, stride=1,
                                  groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.up_illus = nn.ModuleList()
        self.de_illus = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.illumination_down_blocks = nn.ModuleList([])

        ill_chan = latent_dim

        self.illumination_down_blocks.append(
            nn.Conv2d(in_channels=ill_chan, out_channels=chan, kernel_size=4, stride=4, padding=1))
        ill_chan = chan

        self.synthesis = nn.Conv2d(in_channels=chan + ill_chan + ill_chan, out_channels=chan,
                                   kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.middle_blks = nn.Sequential(
                *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.up_illus.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )
            self.de_illus.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        self.fusion = nn.Sequential(
            nn.Conv2d(img_channel * 3, width * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            NONLocalBlock2D(width * 2, sub_sample='bilinear', bn_layer=False),
            nn.Conv2d(width * 2, width * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, 3, 1),
            NONLocalBlock2D(3, sub_sample='bilinear', bn_layer=False)
        )

        self.apply_coeffs = ApplyCoeffs()

    def forward(self, inp, need_reflectance=False, only_reflectance=False, single_frame=None):

        if only_reflectance:  # False
            return self.get_reflectance(inp, single_frame)

        B, N, C, H, W = inp.shape  # torch.Size([16, 5, 3, 256, 256])
        assert N == self.nframes
        inp = self.check_image_size(inp)

        # Construct input
        x_center = inp[:, self.center, :, :, :].contiguous()  # B C H W
        # r, g, b = x_center[:, 0] + 1, x_center[:, 1] + 1, x_center[:, 2] + 1
        # illu_map = (1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.).unsqueeze(1)
        # inverse_illu_map = 1 - illu_map
        # illu_maps = torch.cat([illu_map, illu_map, illu_map], dim=1)
        # inverse_illu_maps = torch.cat([inverse_illu_map, inverse_illu_map, inverse_illu_map], dim=1)
        illumination = self.illumination_module.illumination(x_center, quantize=True)
        illumination_v = self.illumination_module.illumination(1 - x_center, quantize=True)
        # torch.Size([16, 32, 64, 64])

        for block in self.illumination_down_blocks:
            illumination = block(illumination)
            illumination_v = block(illumination_v)
            # Conv2d(32, 256, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
            # illumination: torch.Size([16, 256, 16, 16])
        # ipdb.set_trace()

        # TODO: ACM MM 2024 changes here ############################################################
        # 1. calculate average amplitude
        x_amp_freqs = [torch.fft.rfft2(inp[:, i, :, :, :].contiguous(), norm='backward') for i in range(N)]
        # x_amp_freqs[i]: torch.Size([16, 3, 256, 256])
        x_amp_freq = torch.cat(x_amp_freqs, dim=1)
        # torch.Size([16, 15, 256, 256])
        x_amp_freq_amp = torch.abs(x_amp_freq)
        x_amp_avg = self.amp_map(x_amp_freq_amp)  # torch.Size([16, 3, 256, 256]

        x = []
        aligned_frames = []
        # 2. calculate phase
        for i in range(N):
            x_amp_freqs_pha = torch.angle(x_amp_freqs[i])
            x_amp_pha_new = self.pha_maps[i](x_amp_freqs_pha)
            x_amp_new = x_amp_avg * torch.exp(1j * x_amp_pha_new)
            inp_new = torch.fft.irfft2(x_amp_new, norm='backward').real

        # 3. align frames
        # for i in range(N):
            aligned_x = self.fal[i](inp[:, i, :, :, :], inp[:, self.center, :, :, :], x_amp_avg, inp_new)
            aligned_frames.append(aligned_x)
            # x.append(self.frames_Projection[i](aligned_x))
            # x.append(self.frames_Projection[i](torch.cat([inp_new[i], aligned_x], dim=1)))
            # x.append(self.frames_Projection[i](inp_new + aligned_x))
            x.append(self.frames_Projection[i](aligned_x))
            # x[i]: torch.Size([16, 3, 256, 256])
        # TODO: ACM MM changes here ############################################################

        x = torch.stack(x, dim=1).view(B, N * C, H, W)  # torch.Size([16, 15, 256, 256])

        # Add selective IN here for multi-frame input
        x = self.intro(x)  # self.intro: Conv2d(15, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # torch.Size([16, 16, 256, 256])

        # TODO: change here
        # x = self.attin(x)
        x, x_norm = self.attin(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            # BaselineBlock and down: Conv2d(32, 64, kernel_size=(2, 2), stride=(2, 2))
            x = encoder(x)  # torch.Size([16, 16, 256, 256])  torch.Size([16, 32, 128, 128]) ...
            encs.append(x)
            x = down(x)  # torch.Size([16, 32, 128, 128]) torch.Size([16, 64, 64, 64]) ...
            # torch.Size([16, 128, 32, 32])  torch.Size([16, 256, 16, 16])

        # x is reflectance
        if need_reflectance:  # false
            reflectance = x.clone()

        x = self.synthesis(torch.cat([x, illumination, illumination_v], 1))
        # Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)): torch.Size([16, 256, 16, 16])
        x = self.middle_blks(x)  # BaselineBlock: torch.Size([16, 256, 16, 16])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)  # torch.Size([16, 128, 32, 32])
            x = x + enc_skip
            x = decoder(x)  # torch.Size([16, 128, 32, 32])...torch.Size([16, 16, 256, 256])

        # the illumination decoders are heavy
        cated_illus = torch.cat([illumination, illumination_v], dim=0)
        for decoder, up in zip(self.de_illus, self.up_illus):
            cated_illus = up(cated_illus)
            cated_illus = decoder(cated_illus)

        x = self.ending(x)  # torch.Size([16, 3, 256, 256])
        image_illus = self.end_illu(cated_illus)  # torch.Size([16, 3*4, 256, 256])
        image_illu, image_illu_v = image_illus[:B], image_illus[B:]
        image_illu = self.apply_coeffs(image_illu, x_center)
        image_illu_v = self.apply_coeffs(image_illu_v, x_center)

        brighten_x = self.decomp(x_center, image_illu)
        darken_x = 1 - self.decomp(1 - x_center, image_illu_v)
        fused_x = torch.cat([x, brighten_x, darken_x], dim=1)

        weight_map = self.fusion(fused_x)  # torch.Size([16, 3 * 3, 256, 256])
        w1 = weight_map[:, 0, ...].unsqueeze(1)
        w2 = weight_map[:, 1, ...].unsqueeze(1)
        w3 = weight_map[:, 2, ...].unsqueeze(1)
        out = x * w1 + brighten_x * w2 + darken_x * w3

        # x = x + x_center  # torch.Size([16, 3, 256, 256])

        if need_reflectance:  # no
            return out[:, :, :H, :W], reflectance

        return out[:, :, :H, :W], image_illu, image_illu_v, brighten_x, darken_x, aligned_frames

    def decomp(self, x1, illu_map):
        return x1 / (torch.where(illu_map < x1, x1, illu_map.float()) + 1e-7)

    def get_reflectance(self, inp, frame):
        B, N, C, H, W = inp.shape
        assert N == self.nframes
        for i in range(N):
            inp[:, i, :, :, :] = frame

        inp = self.check_image_size(inp)
        x_center = inp[:, self.center, :, :, :].contiguous()

        x = []
        for i in range(N):
            x.append(self.frames_Projection[i](inp[:, i, :, :, :].contiguous()))

        x = torch.stack(x, dim=1).view(B, N * C, H, W)
        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        return x

    def get_illumination_module(self, lldm_config, vae_config, unet_config, ckpt):
        cfg = yaml.load(open(lldm_config, 'r'), Loader=yaml.Loader)
        cfg_unet = yaml.load(open(unet_config, 'r'), Loader=yaml.Loader)
        cfg_vae = yaml.load(open(vae_config, 'r'), Loader=yaml.Loader)

        vae_model = VQGANLight(**cfg_vae['model'])
        unet = UNetLight(**cfg_unet)

        lldm = LLDM(eps_model=unet, vae_model=vae_model, **cfg)
        load_checkpoint(lldm, ckpt)
        lldm.cuda()

        latent_dim = cfg_vae['model']['latent_dim']

        return latent_dim, lldm

    def check_image_size(self, x):
        _, _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
