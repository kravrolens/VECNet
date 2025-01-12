import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, CharbonnierLoss2, CB_SSIM_Loss, SSIMLoss
from metrics.calculate_PSNR_SSIM import psnr_np
from models.loss_new import FFT_Loss, TV_Loss
import time
import ipdb
from ptflops import get_model_complexity_info

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'cb2':
                self.cri_pix = CharbonnierLoss2().to(self.device)
            elif loss_type == 'cb+ssim':
                self.cri_pix = CB_SSIM_Loss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))

            self.l_pix_w = train_opt['pixel_weight']
            self.lamda = train_opt['reflectance_weight']
            self.l_fft_w = train_opt['fft_weight']
            self.l_tv_w = train_opt['tv_weight']

            self.cri_pix_ill = nn.L1Loss(reduction='sum').to(self.device)
            self.cri_ssim = SSIMLoss().to(self.device)
            self.fft_loss = FFT_Loss().to(self.device)
            self.tv_loss = TV_Loss().to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        # self.fake_H = self.netG(self.var_L)

        # create noise: SDSD code
        img_train_mean = self.var_L.mean(dim=4).mean(dim=3)
        img_train_mean = img_train_mean.unsqueeze(dim=-1).unsqueeze(dim=-1)
        img_train_mean = img_train_mean.expand_as(self.var_L)
        noise = self.var_L - img_train_mean
        input = noise + self.var_L

        self.fake_H, self.illu_map, self.illu_map_v, self.brighten, self.darken, self.aligned_frames = self.netG(
            self.var_L)
        self.fake_H_noise, _, _, _, _, _ = self.netG(input)

        # cb+ssim loss
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        # tv loss
        l_tv = self.l_tv_w * (self.tv_loss(self.illu_map) + self.tv_loss(self.illu_map_v))
        # noise loss
        l_noise = self.l_pix_w * self.cri_pix(self.fake_H_noise, self.real_H)

        l_total = l_pix + l_tv + l_noise
        
        l_total.backward()
        self.optimizer_G.step()

        psnr = psnr_np(self.fake_H.detach(), self.real_H.detach())
        self.log_dict['psnr'] = psnr.item()
        self.log_dict['loss_total'] = l_total.item()

    def test(self):
        self.netG.eval()
        with (torch.no_grad()):
            self.fake_H, self.illu_map, self.illu_map_v, \
                self.brighten, self.darken, self.aligned_frames = self.netG(self.var_L)
        self.netG.train()

    def test_visual(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, self.illu_map, self.illu_map_v, \
                self.brighten, self.darken, self.aligned_frames = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True, need_illu=False):
        out_dict = OrderedDict()

        # choose the first image for each batch, Take one frame from every batch
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        out_dict['aligned_frames'] = [self.aligned_frames[i].detach()[0].float().cpu() for i in
                                      range(len(self.aligned_frames))]

        if need_illu:
            out_dict['illu_map'] = self.illu_map.detach()[0].float().cpu()
            out_dict['illu_map_v'] = self.illu_map_v.detach()[0].float().cpu()
            out_dict['brighten'] = self.brighten.detach()[0].float().cpu()
            out_dict['darken'] = self.darken.detach()[0].float().cpu()
            # ipdb.set_trace()
            del self.illu_map
            del self.illu_map_v
            del self.brighten
            del self.darken

        del self.real_H
        del self.var_L
        del self.fake_H
        del self.aligned_frames

        torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def save_best(self, name):
        self.save_network(self.netG, 'best' + name, 0)
