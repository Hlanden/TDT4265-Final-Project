import albumentations as aug
import cv2
from albumentations.core.transforms_interface import DualTransform

class Resize(DualTransform):
    """Resize the input to the given height and width.
    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, fx=1, fy=0.5, always_apply=False, p=1):
        super(Resize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.fx = fx
        self.fy = fy

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        return cv2.resize(img, dsize=(self.height, self.width), fx=self.fx, fy=self.fy, interpolation=interpolation) 

def build_transforms(cfg, 
                     is_plotting=False,
                     is_train=True):

    trans_list = []
    if cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.ENABLE:
        si = cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.SIZE
        pr = cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.PROBABILITY
        #trans_list.append(aug.augmentations.Resize(si, si, interpolation=cv2.INTER_LINEAR, always_apply=False, p=pr))
        trans_list.append(Resize(si, si, interpolation=cv2.INTER_LINEAR, always_apply=False, p=pr))
    if not is_plotting and is_train:
        if cfg.PREPROCESSING.HORIZONTALFLIP.ENABLE:
            pr = cfg.PREPROCESSING.HORIZONTALFLIP.PROBABILITY 
            trans_list.append(aug.augmentations.transforms.HorizontalFlip(p=pr))

        if cfg.PREPROCESSING.GAUSSIANSMOOTH.ENABLE:
            bl = cfg.PREPROCESSING.GAUSSIANSMOOTH.BLURLIMIT
            sl = cfg.PREPROCESSING.GAUSSIANSMOOTH.SIGMALIMIT
            pr = cfg.PREPROCESSING.GAUSSIANSMOOTH.PROBABILITY

            trans_list.append(aug.augmentations.transforms.GaussianBlur(blur_limit=bl, sigma_limit = sl, p=pr)) 
    
    final_transform = aug.Compose(trans_list, additional_targets={'gt': 'image'})
    
    return final_transform
