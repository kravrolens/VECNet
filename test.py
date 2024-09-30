import cv2

import os.path as osp
import logging
import argparse

import ipdb
from tqdm import tqdm

import options.base_options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
from metrics.calculate_PSNR_SSIM import calculate_psnr, calculate_ssim

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main(write_imgs=False):
    # save_folder = './results/{}_{}'.format(opt['name'], util.get_timestamp())  # save results to folder
    save_folder = '{}/{}_{}'.format(opt['path']['root'], opt['name'], util.get_timestamp())  # save results to folder
    output_folder = osp.join(save_folder, 'images/outputs')
    concated_folder = osp.join(save_folder, 'images/concated')
    util.mkdirs(save_folder)
    util.mkdirs(output_folder)
    util.mkdirs(concated_folder)

    logger = logging.getLogger('base')
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    # opt['path']['pretrain_model_G'] = path_pretrain_model_G
    model = create_model(opt)

    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        psnr_rlt = {}
        psnr_rlt_avg = {}
        psnr_total_avg = 0.
        psnr_over_avg = 0.
        psnr_under_avg = 0.

        ssim_rlt = {}
        ssim_rlt_avg = {}
        ssim_total_avg = 0.
        ssim_over_avg = 0.
        ssim_under_avg = 0.

        over = ['091', '110', '111', '112', '113', '114', '116', '117', '118', '119']

        for val_data in tqdm(val_loader):
            folder = val_data['folder'][0]

            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []
            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []

            idx_d = val_data['idx']
            model.feed_data(val_data)
            model.test()

            visuals = model.get_current_visuals()
            rlt_img = util.tensor2img(visuals['rlt'])
            gt_img = util.tensor2img(visuals['GT'])
            # ipdb.set_trace()

            psnr = calculate_psnr(rlt_img, gt_img)
            psnr_rlt[folder].append(psnr)
            ssim = calculate_ssim(rlt_img, gt_img)
            ssim_rlt[folder].append(ssim)

            mid_ix = dataset_opt['N_frames'] // 2
            input_img = util.tensor2img(visuals['LQ'][mid_ix])  # visuals['LQ']: torch.Size([N, 3, 512, 960])
            # input_img = util.tensor2img(val_data['LQs'][:, mid_ix])  # torch.Size([3, 256, 256])

            # the tag is too long
            # ipdb.set_trace()
            # TODO: fix the tag
            tag = '{}_{}'.format(val_data['folder'][0], idx_d[0].replace('/', '_'))

            if write_imgs:
                cat_img = cv2.hconcat([input_img, rlt_img, gt_img])
                cv2.imwrite(osp.join(output_folder, '{}.png'.format(tag)), rlt_img)
                cv2.imwrite(osp.join(concated_folder, '{}.png'.format(tag)), cat_img)

        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]
            if k in over:
                psnr_over_avg += psnr_rlt_avg[k]
            else:
                psnr_under_avg += psnr_rlt_avg[k]
        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]
            if k in over:
                ssim_over_avg += ssim_rlt_avg[k]
            else:
                ssim_under_avg += ssim_rlt_avg[k]

        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        psnr_over_avg /= len(over)
        ssim_over_avg /= len(over)
        psnr_under_avg /= (len(psnr_rlt) - len(over))
        ssim_under_avg /= (len(ssim_rlt) - len(over))
        # print(psnr_total_avg, ssim_total_avg)
        logger.info('# Test # PSNR-over: {:.4f}:'.format(psnr_over_avg))
        logger.info('# Test # PSNR-under: {:.4f}:'.format(psnr_under_avg))
        logger.info('# Test # SSIM-over: {:.4f}:'.format(ssim_over_avg))
        logger.info('# Test # SSIM-under: {:.4f}:'.format(ssim_under_avg))

        log_s1 = '# Test # PSNR: {:.4f}:'.format(psnr_total_avg)
        log_s2 = '# Test # SSIM: {:.4f}:'.format(ssim_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s1 += ' {}: {:.4f}'.format(k, v)
        for k, v in ssim_rlt_avg.items():
            log_s2 += ' {}: {:.4f}'.format(k, v)
        logger.info(log_s1)
        logger.info(log_s2)
        return psnr_total_avg, ssim_total_avg


if __name__ == '__main__':
    psnr, ssim = main(True)
    # psnr, ssim = main()

    print("psnr: ", psnr)
    print("ssim: ", ssim)

