import ipdb
import yaml

from models.LAN.NAFNet import *
from models.LAN.lldm.vqvae import VQGANLight
from models.LAN.lldm.unet import UNetLight
from models.LAN.lldm import LLDM
from utils.util import load_checkpoint


class BaselinewLLDM(nn.Module):

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
            [framesBlock(channels=[32, 16, 16, 3], block=ConvBlock) for i in range(nframes)])

        self.intro = nn.Conv2d(in_channels=nframes * img_channel, out_channels=width, kernel_size=3, padding=1,
                               stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

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
        # for i in range(down_blocks):
        #     self.illumination_down_blocks.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels=ill_chan, out_channels=ill_chan*4, kernel_size=4, stride=2, padding=1),
        #             nn.LeakyReLU(0.2, True),
        #             ResidualBlock(c*4, c*4)
        #         )
        #     )
        #     ill_chan = ill_chan*4

        self.synthesis = nn.Conv2d(in_channels=chan + ill_chan, out_channels=chan, kernel_size=3, padding=1, stride=1,
                                   groups=1, bias=True)

        self.middle_blks = \
            nn.Sequential(
                *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
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

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, need_reflectance=False, only_reflectance=False, single_frame=None):
        # ipdb.set_trace()
        if only_reflectance:  # False
            return self.get_reflectance(inp, single_frame)

        B, N, C, H, W = inp.shape  # torch.Size([16, 5, 3, 256, 256])
        assert N == self.nframes
        inp = self.check_image_size(inp)

        x_center = inp[:, self.center, :, :, :].contiguous()

        illumination = self.illumination_module.illumination(x_center, quantize=True)
        # torch.Size([16, 32, 64, 64])
        for block in self.illumination_down_blocks:
            illumination = block(illumination)
            # Conv2d(32, 256, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
            # illumination: torch.Size([16, 256, 16, 16])

        x = []
        for i in range(N):
            x.append(self.frames_Projection[i](inp[:, i, :, :, :].contiguous()))
            # x[i]: torch.Size([16, 3, 256, 256])
        x = torch.stack(x, dim=1).view(B, N * C, H, W)  # torch.Size([16, 15, 256, 256])
        x = self.intro(x)  # self.intro: Conv2d(15, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # torch.Size([16, 16, 256, 256])

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

        x = self.synthesis(torch.cat([x, illumination], 1))
        # Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)): torch.Size([16, 256, 16, 16])
        x = self.middle_blks(x)  # BaselineBlock: torch.Size([16, 256, 16, 16])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)  # torch.Size([16, 128, 32, 32])
            x = x + enc_skip
            x = decoder(x)  # torch.Size([16, 128, 32, 32])...torch.Size([16, 16, 256, 256])

        x = self.ending(x)  # torch.Size([16, 3, 256, 256])
        x = x + x_center   # torch.Size([16, 3, 256, 256])

        if need_reflectance:  # no
            return x[:, :, :H, :W], reflectance

        return x[:, :, :H, :W]

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
