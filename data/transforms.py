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


    transform = aug.Compose([
        aug.augmentations.Resize(384, 384, interpolation=INTER_LINEAR, always_apply=False, p=1), 
        aug.augmentations.transforms.HorizontalFlip(p=1),
        aug.augmentations.transforms.GaussianBlur(blur_limit=111, sigma_limit = 0, p=1) 
    ])


    final_transform = aug.Compose(transforms)
    
    return transform