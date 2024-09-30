import argparse
import random
import logging

import ipdb
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp

import options.base_options as option
from metrics.calculate_PSNR_SSIM import *
from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr

from data.data_sampler import DistIterSampler
from data import create_dataloader, create_dataset
from models import create_model
from utils import util


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default=None)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        # if resume_state is None:
        #     util.mkdir_and_rename(
        #         opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        # logger.info(option.dict2str(opt))

        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            # tb_logger = SummaryWriter(log_dir='./tb_logger/' + opt['name'])
            tb_logger = SummaryWriter(log_dir=(os.path.join(opt['path']['root'], 'tb_logger',
                                                            opt['name'] + '_' + util.get_timestamp())))
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    dataset_ratio = 1  # 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))

            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                # no distributed training
                train_sampler = None

            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        # else:
        #     raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    # create model
    model = create_model(opt)

    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        # start_epoch = 72
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        del resume_state
    else:
        current_step = 0
        start_epoch = 0

    # initialized metrics values here
    best_psnr_avg = 0
    best_step_psnr = 0
    best_ssim_avg = 0
    best_step_ssim = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))

    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        # initialized metrics values here for each epoch
        total_psnr = 0.
        total_loss = 0.
        # print_iter = 0

        for _, train_data in enumerate(tqdm(train_loader)):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            # print('path_GT & LQ: ', train_data['path_GT'], train_data['path_LQ'])
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # visualizations
            # visuals = model.get_current_visuals()
            # rlt_img = util.tensor2img(visuals['rlt'])
            # gt_img = util.tensor2img(visuals['GT'])
            # tag = '{}.{}'.format(train_data['folder'], train_data['idx'][0].replace('/', '-'))
            # # print(osp.join(output_folder, '{}.png'.format(tag)))
            # cv2.imwrite(os.path.join('./results/out', '{}.png'.format(tag)), rlt_img)
            # cv2.imwrite(os.path.join('./results/gt', '{}.png'.format(tag)), gt_img)

            # ipdb.set_trace()
            # logs
            # print more logs to tensorboard and print here, such as loss, metrics, etc.
            if current_step % opt['logger']['print_freq'] == 0:
                # print_iter += 1
                logs = model.get_current_log()
                message = 'epoch:{:3d}, iter:{:8d},'.format(epoch, current_step)
                # message = 'epoch:{:3d}, lr:'.format(epoch)
                # for v in model.get_current_learning_rate():
                #     message += '{:.3e},'.format(v)

                # total_loss += logs['loss_total']
                # total_psnr += logs['psnr']
                # mean_total = total_loss / print_iter
                # mean_psnr = total_psnr / print_iter

                # TODO: why the mean_psnr is so small?
                mean_total = logs['loss_total']
                mean_psnr = logs['psnr']

                message += ' {:s}: {:.4e}'.format('loss', mean_total)
                message += ' {:s}: {:.4f}'.format('psnr', mean_psnr)

                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    if rank <= 0:
                        tb_logger.add_scalar('train_psnr', mean_psnr, current_step)
                        tb_logger.add_scalar('train_loss', mean_total, current_step)
                if rank <= 0:
                    logger.info(message)

                # for k, v in logs.items():
                #     message += '{:s}: {:.4e} '.format(k, v)
                #     # tensorboard logger
                #     if opt['use_tb_logger'] and 'debug' not in opt['name']:
                #         if rank <= 0:
                #             tb_logger.add_scalar(k, v, current_step)
                # if rank <= 0:
                #     logger.info(message)

            # validation
            # check how to calc metrics and save images
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['dist']:
                    # multi-GPU testing
                    psnr_rlt = {}  # with border and center frames
                    ssim_rlt = {}  # with border and center frames

                    random_index = random.randint(0, len(val_set) - 1)
                    for idx in range(rank, len(val_set), world_size):

                        if not (idx == random_index):
                            continue

                        val_data = val_set[idx]
                        val_data['LQs'].unsqueeze_(0)
                        val_data['GT'].unsqueeze_(0)
                        folder = val_data['folder']
                        idx_d, max_idx = val_data['idx'].split('/')
                        idx_d, max_idx = int(idx_d), int(max_idx)
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                            ssim_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                        model.feed_data(val_data)
                        model.test()
                        visuals = model.get_current_visuals()
                        src_img = util.tensor2img(visuals['LQ'][2])
                        rlt_img = util.tensor2img(visuals['rlt'])  # the enhanced frame
                        gt_img = util.tensor2img(visuals['GT'])  # the ground truth

                        save_img = np.concatenate([src_img, rlt_img, gt_img], axis=0)
                        im_path = os.path.join(opt['path']['val_images'], '%06d.png' % current_step)
                        cv2.imwrite(im_path, save_img.astype(np.uint8))

                        # calculate PSNR
                        psnr_inst = calc_psnr(gt_img, rlt_img, data_range=1.)
                        ssim_inst = calc_ssim(gt_img, rlt_img, multichannel=True)
                        if math.isinf(psnr_inst) or math.isnan(psnr_inst):
                            psnr_inst = 0.
                            ssim_inst = 0.
                        psnr_rlt[folder][idx_d] = psnr_inst
                        ssim_rlt[folder][idx_d] = ssim_inst

                        # psnr_rlt[folder][idx_d] = calculate_psnr(rlt_img, gt_img)
                        # ssim_rlt[folder][idx_d] = calculate_ssim(rlt_img, gt_img)

                    # collect data
                    for _, v in psnr_rlt.items():
                        dist.reduce(v, 0)
                    for _, v in ssim_rlt.items():
                        dist.reduce(v, 0)
                    dist.barrier()

                    if rank == 0:
                        psnr_rlt_avg = {}
                        ssim_rlt_avg = {}
                        psnr_total_avg = 0.
                        ssim_total_avg = 0.
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                            psnr_total_avg += psnr_rlt_avg[k]
                        for k, v in ssim_rlt.items():
                            ssim_rlt_avg[k] = torch.mean(v).cpu().item()
                            ssim_total_avg += ssim_rlt_avg[k]
                        psnr_total_avg /= len(psnr_rlt)
                        ssim_total_avg /= len(ssim_rlt)
                        log_s1 = '# Validation # PSNR: {:.4f}:'.format(psnr_total_avg)
                        log_s2 = '# Validation # SSIM: {:.4f}:'.format(ssim_total_avg)

                        for k, v in psnr_rlt_avg.items():
                            log_s1 += ' {}: {:.4f}'.format(k, v)
                        for k, v in ssim_rlt_avg.items():
                            log_s2 += ' {}: {:.4f}'.format(k, v)
                        logger.info(log_s1)
                        logger.info(log_s2)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('valid_psnr_avg', psnr_total_avg, current_step)
                            tb_logger.add_scalar('valid_ssim_avg', psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)
                            for k, v in ssim_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)
                else:
                    psnr_rlt = {}  # with border and center frames
                    ssim_rlt = {}
                    psnr_rlt_avg = {}
                    ssim_rlt_avg = {}
                    psnr_total_avg = 0.
                    ssim_total_avg = 0.

                    for val_data in tqdm(val_loader):
                        folder = val_data['folder'][0]
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = []
                            ssim_rlt[folder] = []

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8

                        # calculate PSNR and SSIM
                        psnr = calculate_psnr(rlt_img, gt_img)
                        ssim = calculate_ssim(rlt_img, gt_img)
                        psnr_rlt[folder].append(psnr)
                        ssim_rlt[folder].append(ssim)
                    for k, v in psnr_rlt.items():
                        psnr_rlt_avg[k] = sum(v) / len(v)
                        psnr_total_avg += psnr_rlt_avg[k]
                    for k, v in ssim_rlt.items():
                        ssim_rlt_avg[k] = sum(v) / len(v)
                        ssim_total_avg += ssim_rlt_avg[k]
                    psnr_total_avg /= len(psnr_rlt)
                    ssim_total_avg /= len(ssim_rlt)

                    # average score and each video score
                    log_s1 = '# Validation # PSNR: {:.4f}'.format(psnr_total_avg)
                    log_s2 = '# Validation # SSIM: {:.4f}'.format(ssim_total_avg)

                    # each video score
                    for k, v in psnr_rlt_avg.items():
                        log_s1 += ' ID-{}:{:.4f}'.format(k, v)
                    for k, v in ssim_rlt_avg.items():
                        log_s2 += ' ID-{}:{:.4f}'.format(k, v)

                    logger.info(log_s1)
                    logger.info(log_s2)
                    logger.info('Previous best AVG-PSNR: {:.4f} Previous best AVG-step: {}'.
                                format(best_psnr_avg, best_step_psnr))
                    logger.info('Previous best AVG-SSIM: {:.4f} Previous best AVG-step: {}'.
                                format(best_ssim_avg, best_step_ssim))

                    if psnr_total_avg > best_psnr_avg:
                        best_psnr_avg = psnr_total_avg
                        best_step_psnr = current_step
                        logger.info('Saving best average models!!!!!!!The BEST PSNR is:{:4f}'.format(best_psnr_avg))
                        model.save_best('avg_psnr')

                    if ssim_total_avg > best_ssim_avg:
                        best_ssim_avg = ssim_total_avg
                        best_step_ssim = current_step
                        logger.info('Saving best average models!!!!!!!The BEST SSIM is:{:4f}'.format(best_ssim_avg))
                        model.save_best('avg_ssim')

                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('valid_psnr_avg', psnr_total_avg, current_step)
                        tb_logger.add_scalar('valid_ssim_avg', ssim_total_avg, current_step)
                        for k, v in psnr_rlt_avg.items():
                            tb_logger.add_scalar(k, v, current_step)
                        for k, v in ssim_rlt_avg.items():
                            tb_logger.add_scalar(k, v, current_step)

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()
