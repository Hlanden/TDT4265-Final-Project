import albumentations as aug
from albumentations.core.composition import set_always_apply
import cv2
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
#from albumentations.augmentations.crops.functional import random_crop
from albumentations.augmentations.functional import random_crop
import random

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


class RandomCrop(DualTransform):
    """Crop a random part of the input.
    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, height_scale, width_scale, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height_scale = height_scale
        self.width_scale = width_scale

    def apply(self, img, h_start=0, w_start=0, **params):
        x, y = img.shape

        #MERK: Kan hende x og y må bytte plass her
        height = int(x*self.height_scale)
        width = int(y*self.width_scale )
        return random_crop(img, height, width, h_start, w_start)

    def get_params(self):
        return {"h_start": random.random(), "w_start": random.random()}

    def get_transform_init_args_names(self):
        return ("height", "width")

class Padding(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Padding, self).__init__(always_apply, p)

    def apply(self, img, **params):
        height, width = img.shape
        top = bottom = int((width - height)/2) if width > height else 0 
        right = left = -int((width - height)/2) if width < height else 0 
        if height + 2*top < width + 2*right:
            top += 1
        elif height + 2*top > width + 2*right:
            right += 1
        return cv2.copyMakeBorder(img, top, bottom, right, left, cv2.BORDER_CONSTANT)

def build_transforms(cfg, 
                     is_train=True,
                     tee=False):

    image_and_gt_transform = [] #train transforms
    only_img_list = [] #kun img blir endret
    
    #val_trans_list = [] #bilde og gt blir endret
    if cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.ENABLE and not tee:
        if cfg.PREPROCESSING.RESIZE.FX:
            fx_num = 0.154/cfg.PREPROCESSING.RESIZE.FX
            fy_num = 0.308/cfg.PREPROCESSING.RESIZE.FY    
        else:
            fx_num=0
            fy_num=0
        x = cfg.PREPROCESSING.RESIZE.X
        y = cfg.PREPROCESSING.RESIZE.Y

        image_and_gt_transform.append(Resize(x, y, fx=fx_num, fy=fy_num, interpolation=cv2.INTER_LINEAR, p=1))

    if cfg.PREPROCESSING.NORMALIZE.ENABLE:
        m = cfg.PREPROCESSING.NORMALIZE.MEAN
        s = cfg.PREPROCESSING.NORMALIZE.STD

        only_img_list.append(aug.augmentations.transforms.Normalize(mean= m, std = s, max_pixel_value=1.0, always_apply=False, p=1.0))
    
    if cfg.PREPROCESSING.GAUSSIANSMOOTH.ENABLE:
        bl = cfg.PREPROCESSING.GAUSSIANSMOOTH.BLURLIMIT
        sl = cfg.PREPROCESSING.GAUSSIANSMOOTH.SIGMALIMIT
        pr = cfg.PREPROCESSING.GAUSSIANSMOOTH.PROBABILITY

        only_img_list.append(aug.augmentations.transforms.GaussianBlur(blur_limit=bl, sigma_limit = sl, p=pr)) 
    
    if is_train:
        if cfg.PREPROCESSING.ELASTICDEFORM.ENABLE:
            a = cfg.PREPROCESSING.ELASTICDEFORM.ALPHA
            sig = cfg.PREPROCESSING.ELASTICDEFORM.SIGMA
            a_af = cfg.PREPROCESSING.ELASTICDEFORM.ALPHA_AFFINE
            pr = cfg.PREPROCESSING.ELASTICDEFORM.PROBABILITY

            image_and_gt_transform.append(aug.augmentations.transforms.ElasticTransform(alpha=a, sigma=sig, alpha_affine=a_af, interpolation=1, border_mode=1, always_apply=False, p=pr))
        
        
        if cfg.PREPROCESSING.GRIDDISTORTIAN.ENABLE:
            steps = cfg.PREPROCESSING.GRIDDISTORTIAN.MUM_STEPS
            dis_lim = cfg.PREPROCESSING.GRIDDISTORTIAN.DISTORT_LIMIT
            pr = cfg.PREPROCESSING.GRIDDISTORTIAN.PROB

            image_and_gt_transform.append(aug.augmentations.transforms.GridDistortion(num_steps=steps, distort_limit= dis_lim, interpolation=1, border_mode=0, p = pr))

        if cfg.PREPROCESSING.RANDOMCROP.ENABLE:
            x = cfg.PREPROCESSING.RANDOMCROP.X_RESIZE
            y = cfg.PREPROCESSING.RANDOMCROP.Y_RESIZE
            pr = cfg.PREPROCESSING.RANDOMCROP.PROB

            image_and_gt_transform.append(RandomCrop(height_scale = x, width_scale = y, always_apply=False, p=pr))




        if cfg.PREPROCESSING.ROTATE.ENABLE:
            lim =  cfg.PREPROCESSING.ROTATE.LIMIT  
            r_mod = cfg.PREPROCESSING.ROTATE.BORDER_MODE
            pr = cfg.PREPROCESSING.ROTATE.PROB

            image_and_gt_transform.append(aug.augmentations.transforms.Rotate(limit = lim, border_mode= r_mod, p= pr))
        if cfg.PREPROCESSING.HORIZONTALFLIP.ENABLE:
            pr = cfg.PREPROCESSING.HORIZONTALFLIP.PROBABILITY 
            image_and_gt_transform.append(aug.augmentations.transforms.HorizontalFlip(p=pr))

       
    if tee:
        pass


    final_transform = aug.Compose(image_and_gt_transform, additional_targets={'gt': 'image'})
    only_img_transform = aug.Compose(only_img_list)
    return final_transform, only_img_transform

