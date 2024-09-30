import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class FFT_Loss(nn.Module):
    def __init__(self):
        super(FFT_Loss, self).__init__()

    def forward(self, x, gt):
        x = x + 1e-8
        gt = gt + 1e-8
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_amp = torch.abs(x_freq)
        x_phase = torch.angle(x_freq)

        gt_freq = torch.fft.rfft2(gt, norm='backward')
        gt_amp = torch.abs(gt_freq)
        gt_phase = torch.angle(gt_freq)

        loss_amp = torch.mean(torch.sum((x_amp - gt_amp) ** 2))
        loss_phase = torch.mean(torch.sum((x_phase - gt_phase) ** 2))
        return loss_amp, loss_phase


class AMPLoss(nn.Module):
    def __init__(self):
        super(AMPLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.abs(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.abs(y)

        return self.cri(x_mag, y_mag)


class TV_Loss(nn.Module):
    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class ICloss(torch.nn.Module):
    # illumination contrastive loss

    def __init__(self, k=5):
        super(ICloss, self).__init__()

        self.D = nn.L1Loss()
        self.k = k  # number of negatives, same as number of frames
        vgg16_model = vgg16(pretrained=True)
        self.color_extractor = vgg16_model.features[:23]

    def forward(self, pred, gt_images, lq_images):
        '''
        lq_images      (B, N, 3, H, W):     input to the network
        pred         (B, 3, H, W):        output of the network
        gt_images    (B, N, 3, H, W):     ground truth image
        '''

        B, N, C, H, W = gt_images.shape[0], gt_images.shape[1], gt_images.shape[2], gt_images.shape[3], gt_images.shape[4]

        # calculate style representations
        self.SR_extractor = self.color_extractor.to(pred.get_device())
        self.SR_extractor.eval()

        # ipdb.set_trace() 需要先要降低分辨率
        pooled_pred = F.max_pool2d(pred, (4, 4))
        pooled_gt_image = F.max_pool2d(gt_images.view(B, -1, H, W), (4, 4))
        pooled_lq_image = F.max_pool2d(lq_images.view(B, -1, H, W), (4, 4))

        # choose the negative samples from the same batch
        h_pred = self.SR_extractor(pooled_pred)  # cares about over-under exposure area
        h_plus = self.SR_extractor(pooled_gt_image).view(B, N, C, H, W)  # (N, 512, 32, 32)  48*48
        h_minus = self.SR_extractor(pooled_lq_image).view(B, N, C, H, W)

        sum_negatives = self.D(h_pred, h_minus[:, 0]) + self.D(h_pred, h_minus[:, 1]) + self.D(h_pred, h_minus[:, 2]) + \
                        self.D(h_pred, h_minus[:, 3]) + self.D(h_pred, h_minus[:, 4]) + 1e-8

        l_ss_cr = self.D(h_pred, h_plus[:, 0]) / (self.D(h_pred, h_plus[:, 0]) + sum_negatives) + \
                  self.D(h_pred, h_plus[:, 1]) / (self.D(h_pred, h_plus[:, 1]) + sum_negatives) + \
                  self.D(h_pred, h_plus[:, 2]) / (self.D(h_pred, h_plus[:, 2]) + sum_negatives) + \
                  self.D(h_pred, h_plus[:, 3]) / (self.D(h_pred, h_plus[:, 3]) + sum_negatives) + \
                  self.D(h_pred, h_plus[:, 4]) / (self.D(h_pred, h_plus[:, 4]) + sum_negatives)

        return l_ss_cr

