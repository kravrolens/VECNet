import torch
import torch.nn as nn
import einops
from einops import rearrange

from models.LAN.lldm.layers import ResidualBlock, Attention


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings: int, latent_dim: int):
        """
        Vector quantizer that discretizes the continuous latent z. Adapted from
        https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py.
        Args:
            n_embeddings (int): Codebook size
            latent_dim (int): Dimension of the latent z (channels)
        """
        super(VectorQuantizer, self).__init__()

        self.n_emb = n_embeddings
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.n_emb, self.latent_dim)
        self.embedding.weight.data.uniform_(-1. / self.latent_dim, 1. / self.latent_dim)

    def forward(self, z: torch.Tensor):
        """
        Maps the output of the encoder network z (continuous) to a discrete one-hot
        vector z_q, where the index indicates the closest embedding vector e_j. The
        latent z is detached as first step to allow straight through backprop.
        Args:
            z: Output of the encoder network, shape [bs, latent_dim, h, w]
        Returns:
            z_q: Quantized z
        """
        bs, c, h, w = z.shape

        # flatten input from [bs, c, h, w] to [bs*h*w, c]
        z_flat = einops.rearrange(z, 'b c h w -> (b h w) c')

        # calculate distances between each z [bs*h*w, c]
        # and e_j [n_emb, c]: (z - e_j)² = z² + e² - e*z*2
        z_sq = torch.sum(z_flat**2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding.weight**2, dim=1)
        e_z = torch.matmul(z_flat, self.embedding.weight.t())
        distances = z_sq + e_sq - 2 * e_z    # [bs*h*w, n_emb]

        # get index of the closest embedding e_j for each vector z
        argmin_inds = torch.argmin(distances, dim=1)

        # one-hot encode
        argmin_one_hot = nn.functional.one_hot(argmin_inds, num_classes=self.n_emb).float().to(z.device)

        # multiply one-hot w. embedding weights to get quantized z
        z_q = torch.matmul(argmin_one_hot, self.embedding.weight)

        # reshape back to [bs, c, h, w]
        z_q = einops.rearrange(z_q, '(b h w) c -> b c h w', b=bs, h=h, w=w)

        return z_q

class EncoderLight(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 channels = None,
                 dim_keys: int = 64, n_heads: int = 4):
        """
        Lightweight encoder for VQ-GAN with less parameters. The final
        latent resolution will be: img_size / 2^{len(channels)}.

        Args:
            in_channels: Number of input channels of the image
            latent_dim: Number of channels for the latent space
            channels: List of channels for the number of down/up steps, eg [16, 32, 64]
            dim_keys: Dimension of keys, queries, values for attention layers
            n_heads: Number of heads for multi-head attention
        """
        super().__init__()

        self.channels = channels if channels is not None else [32, 64]
        self.n_blocks = len(self.channels)

        # initial convolutional layer
        self.down_blocks = nn.ModuleList([])
        prev_channel = in_channels
        for c in self.channels:
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_channel, c, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True),
                    ResidualBlock(c, c)
                )
            )
            prev_channel = c

        # bottleneck
        self.mid_attn = Attention(self.channels[-1], dim_keys, n_heads)
        self.mid_block = ResidualBlock(self.channels[-1], self.channels[-1])

        # output
        self.out = nn.Conv2d(self.channels[-1], latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        for block in self.down_blocks:
            x = block(x)

        # bottleneck
        x = self.mid_attn(x)
        x = self.mid_block(x)

        x = self.out(x)

        return x

class DecoderLight(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 channels = None,
                 dim_keys: int = 64, n_heads: int = 4):
        """
        Lightweight decoder for VQ-GAN with less parameters, converting
        a latent representation back to an image.

        Args:
            in_channels: Number of input channels of the image
            latent_dim: Number of channels for the latent space
            channels: List of channels for the number of down/up steps, eg [16, 32, 64]. Note
                that for the decoder the channels list will be reversed
            dim_keys: Dimension of keys, queries, values for attention layers
            n_heads: Number of heads for multi-head attention
        """
        super().__init__()
        self.channels = channels if channels is not None else [32, 64]
        self.n_blocks = len(self.channels)

        self.init_conv = nn.Sequential(
            nn.Conv2d(latent_dim, self.channels[-1], kernel_size=3, padding=1),
            nn.SiLU()
        )

        # bottleneck
        self.mid_block = ResidualBlock(self.channels[-1], self.channels[-1])
        self.mid_attn = Attention(self.channels[-1], dim_keys, n_heads)

        # decoder
        self.up_blocks = nn.ModuleList([])
        prev_channel = self.channels[-1]
        for c in reversed(self.channels):
            self.up_blocks.append(
                nn.Sequential(
                    ResidualBlock(prev_channel, c),
                    nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)
                )
            )
            prev_channel = c

        # output
        self.out = nn.Conv2d(self.channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, features=False):
        # bottleneck
        x = self.init_conv(x)
        x = self.mid_block(x)
        x = self.mid_attn(x)

        # torch.Size([1, 256, 32, 60])
        # torch.Size([1, 128, 64, 120])
        # torch.Size([1, 64, 128, 240])
        # torch.Size([1, 32, 256, 480])

        # torch.Size([1, 32, 64, 120])
        # torch.Size([1, 64, 64, 120])
        # torch.Size([1, 64, 128, 240])
        # torch.Size([1, 32, 256, 480])
        for block in self.up_blocks:
            x = block(x)

        x = torch.tanh(self.out(x))
        return x
        
class VQGANLight(nn.Module):
    def __init__(self, latent_dim: int, autoencoder_cfg: dict, n_embeddings: int = 512):
        """
        Lightweight Vector-quantized GAN (paper: https://arxiv.org/abs/2012.09841)
        with fewer parameters in the encoder/decoder.

        Args:
            latent_dim: Latent dimension of the embedding/codebook
            autoencoder_cfg: Dictionary containing the information for the encoder and decoder. For
                example {'in_channels': 3, 'channels': [16, 32, 64], 'dim_keys': 64, 'n_heads': 4}.
            n_embeddings: Number of embeddings for the codebook
        """
        super().__init__()
        self.encoder = EncoderLight(latent_dim=latent_dim, **autoencoder_cfg)
        self.vq = VectorQuantizer(n_embeddings, latent_dim)
        self.decoder = DecoderLight(latent_dim=latent_dim, **autoencoder_cfg)

    def forward(self, x: torch.Tensor):
        """ Forward pass through vector-quantized variational autoencoder.

        Args:
            x: Input image tensor.
        Returns:
            x_hat: Reconstructed image x
            z_e: Latent (un-quantized) representation of image x
            z_q: Quantized latent representation of image x
        """
        z_e = self.encoder(x)
        z_q = self.vq(z_e)

        # preserve gradients
        z_q_ = z_e + (z_q - z_e).detach()
        x_hat = self.decoder(z_q_)

        return x_hat, z_e, z_q

    def encode(self, x: torch.Tensor):
        """ Encode input image.

        Args:
            x: Input image
        Returns:
            z_e: Encoded input image (un-quantized).
        """
        z_e = self.encoder(x)

        return z_e

    def quantize(self, z_e: torch.Tensor):
        """ Quantize latent representation.

        Args:
            z_e: Un-quantized latent representation (encoded image).
        Returns:
            z_q: Quantized embedding.
        """
        z_q = self.vq(z_e)

        return z_q

    def decode(self, z_e: torch.Tensor, features=False):
        """ Decode latent representation to input image.

        Args:
            z_e: Un-quantized latent representation.
        Returns:
            x_hat: Reconstructed input image.
        """
        z_q = self.vq(z_e)
        x_hat = self.decoder(z_q, features)

        return x_hat


