from typing import Dict, Tuple

import time
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class LLDM(nn.Module):
    def __init__(
            self,
            eps_model: nn.Module,
            vae_model: nn.Module,
            n_steps: int,
    ):
        super(LLDM, self).__init__()
        self.n_steps = n_steps
        self.eps_model = eps_model
        self.vae_model = vae_model

        # we don't want to train any parameters of the vae model
        for param in vae_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, x: torch.Tensor):

        x = self.vae_model.encode(x)
        x = self.vae_model.quantize(x)

        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor, features=False):

        x_hat = self.vae_model.decode(x, features)

        return x_hat

    def fusion(self, LQ, GT, t):
        alpha = t / self.n_steps
        fusion_img = torch.zeros_like(LQ).cuda()
        for i in range(LQ.shape[0]):
            fusion_img[i] = alpha[i] * GT[i] + (1. - alpha[i]) * LQ[i]
        return fusion_img

    def train_loss(self, LQ, GT, criterion):
        t = torch.randint(0, self.n_steps, (LQ.shape[0],)).cuda()

        x = self.fusion(LQ, GT, t)
        y = self.fusion(LQ, GT, t + 1)

        ld_x = self.encode(x)
        ld_y = self.encode(y)
        y_pred = self.eps_model(ld_x, t)

        loss = criterion(y_pred, ld_y)
        return loss

    @torch.no_grad()
    def enhance(self, x, sample_step=None):
        if sample_step == None:
            sample_step = self.n_steps

        x = self.encode(x)
        for t in range(sample_step):
            x = self.eps_model(x, torch.full((x.shape[0],), t, dtype=torch.long).cuda())
        x = self.decode(x)
        return x

    def illumination(self, x, quantize=False, sample_step=None):
        if sample_step == None:
            sample_step = self.n_steps

        x = self.encode(x)
        for t in range(sample_step):
            x = self.eps_model(x, torch.full((x.shape[0],), t, dtype=torch.long).cuda())

        if quantize:
            features = self.vae_model.quantize(x)
        else:
            features = x
        # features = self.decode(x, features=True)
        return features
