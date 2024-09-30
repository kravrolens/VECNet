from models.archs.Enhance_arch_PDN import BaselinePDN as EnhanceNet

def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'BaselinePDN':
        netG = EnhanceNet(lldm_config=opt_net['lldm_config'],
                           vae_config=opt_net['vae_config'],
                           unet_config=opt_net['unet_config'],
                           lldm_ckpt=opt_net['lldm_ckpt'],
                           frozen_lldm=opt_net['frozen_lldm'],
                           down_blocks=opt_net['down_blocks'],
                           width=opt_net['width'],
                           nframes=opt_net['nframes'],
                           middle_blk_num=opt_net['middle_blk_num'],
                           enc_blk_nums=opt_net['enc_blk_nums'],
                           dec_blk_nums=opt_net['dec_blk_nums'],
                           dw_expand=opt_net['dw_expand'],
                           ffn_expand=opt_net['ffn_expand'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
