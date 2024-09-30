import logging

import ipdb

logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'video_dlan':
        from models.video_model_DLAN import VideoBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    # ipdb.set_trace()
    return m
