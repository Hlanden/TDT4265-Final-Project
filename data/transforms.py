import albumentations as aug


def build_transforms(cfg, is_train=True):

    trans_list = []
    if cfg.cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.ENABLE:
        si = cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.SIZE
        pr = cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.PROBABILITY
        trans_list.append(aug.augmentations.Resize(si, si, interpolation=INTER_LINEAR, always_apply=False, p=pr))

    if cfg.PREPROCESSING.HORIZONTALFLIP.ENABLE:
        pr = cfg.PREPROCESSING.HORIZONTALFLIP.PROBABILITY 
        trans_list.append(aug.augmentations.transforms.HorizontalFlip(p=pr))

    if cfg.PREPROCESSING.GAUSSIANSMOOTH.ENABLE:
        bl = cfg.PREPROCESSING.GAUSSIANSMOOTH.BLURLIMIT
        sl = cfg.PREPROCESSING.GAUSSIANSMOOTH.SIGMALIMIT
        pr = cfg.PREPROCESSING.GAUSSIANSMOOTH.PROBABILITY

        trans_list.append(aug.augmentations.transforms.GaussianBlur(blur_limit=bl, sigma_limit = sl, p=pr)) 

    final_transform = aug.Compose(trans_list)
    
    return final_transform