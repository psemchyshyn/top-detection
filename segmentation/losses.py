import segmentation_models_pytorch as smp


def get_loss(name):
    if name == 'jaccard':
        return smp.losses.JaccardLoss(smp.losses.constants.BINARY_MODE)
    elif name == 'dice':
        return smp.losses.DiceLoss(smp.losses.constants.BINARY_MODE)
    elif name == 'focal':
        return smp.losses.FocalLoss(smp.losses.constants.BINARY_MODE)
    elif name == 'lovasz':
        return smp.losses.LovaszLoss(smp.losses.constants.BINARY_MODE)
    elif name == 'bce':
        return smp.losses.SoftBCEWithLogitsLoss()
    elif name == 'jaccard+focal':
        loss1 = smp.losses.JaccardLoss(smp.losses.constants.BINARY_MODE)
        loss2 = smp.losses.FocalLoss(smp.losses.constants.BINARY_MODE)
        return lambda x, y: loss1(x, y) + loss2(x, y)
    elif name == 'jaccard+bce':
        loss1 = smp.losses.JaccardLoss(smp.losses.constants.BINARY_MODE)
        loss2 = smp.losses.SoftBCEWithLogitsLoss()
        return lambda x, y: loss1(x, y) + loss2(x, y.float())
    else:
        raise (f'No such loss: {name}')
