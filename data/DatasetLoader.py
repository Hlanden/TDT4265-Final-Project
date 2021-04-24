if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.getcwd())
from itertools import tee
from matplotlib.colors import cnames
from matplotlib.pyplot import axes
import numpy as np
from numpy.core import numeric
import torch
from torchvision import transforms, datasets
import albumentations as aug 

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, dataloader, sampler, Subset
from PIL import Image
import os
from medimage.medimage import image
from torchvision.transforms.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as TF
import cv2
from data.transforms import Resize, Padding, build_transforms
from config.defaults import cfg

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self,
                 cfg,
                 transforms=None,
                 tee=False):  # legger til muligheten for transform her

        super().__init__()
        
        if not tee:
            self.dataset_dir = Path(cfg.DATASETS.BASE_PATH + cfg.DATASETS.CAMUS)
        else:
            self.dataset_dir = Path(cfg.DATASETS.BASE_PATH + cfg.DATASETS.TEE)
        print('Loading data from: ', self.dataset_dir)
        self.classes = cfg.MODEL.CLASSES
        self.model_depth = len(cfg.UNETSTRUCTURE.CONTRACTBLOCK)
        self.tee = tee
        self.transforms = transforms
            
            
        
        # TODO: Fix if statement below. Not obvious enough to understand what is happening!
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        if tee:
            gt_dir = Path(str(self.dataset_dir) + '/train_gt')
            gray_dir = Path(str(self.dataset_dir) + '/train_gray')
            self.files = [self.combine_files(f, gt_dir) for f in gray_dir.iterdir() if not f.is_dir()]
        else:
            self.files = []
            for patient in self.dataset_dir.iterdir():
                for f in patient.iterdir():
                    filename, filetype = os.path.splitext(f)

                    if not f.is_dir() and str(f).__contains__('.mhd') \
                        and not filename.__contains__('gt') \
                        and not filename.__contains__('sequence'): #TODO: What shall we do with sequence?
                            self.files.append(self.combine_files(filename, ''))
       

    def combine_files(self, gray_file:Path, gt_file):
        if not self.tee:
            files = {'gray': gray_file + '.mhd', 
                    'gt': gray_file + '_gt.mhd'}

        else:
            gt_file = gt_file/gray_file.name.replace('gray', 'gt_gt')
            gt_file, _ = os.path.splitext(gt_file)
            gt_file += '.tif'
            files = {'gray': gray_file, 
                    'gt': gt_file}
        
        """elif self.camus_resized:
            files = {'gray': gray_file, 
                    'gt': gt_file/gray_file.name.replace('gray', 'gt')}"""

        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        if not self.tee:
            raw_us = image(self.files[idx]['gray']).imdata
        else:
            raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                              ], axis=2) 

    
        if invert:
            raw_us = raw_us.transpose((2,0,1))

        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False, transforms=False):
        #open mask file
        if not self.tee:
            raw_mask = image(self.files[idx]['gt']).imdata
            raw_mask = raw_mask.squeeze()
            combined_classes = np.zeros_like(raw_mask)
            for c in self.classes:
                combined_classes += np.where(raw_mask==c, c, 0).astype(np.uint8)
            raw_mask = combined_classes

        else:               
            raw_mask = np.array(Image.open(self.files[idx]['gt']))
            raw_mask = raw_mask[:,:,1]
            combined_classes = np.zeros_like(raw_mask)
            for c in [1, 2]:
                combined_classes += np.where(raw_mask==int(127.5*c), c, 0).astype(np.uint8)
            raw_mask = combined_classes
            
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    

        

    def __getitem__(self, idx):
        #get the image and mask as arrays

        x = self.open_as_array(idx, invert=True).astype(np.float32)
        y = self.open_mask(idx, add_dims=False).astype(np.float32)
        shape = y.shape
        pad_x = 0
        pad_y = 0
        if self.tee:
            x = np.rot90(x, k=2, axes=(1, 2))
            y = np.rot90(y, k=2)
        
            
        if self.transforms:
            if type(self.transforms) == list:
                aug_data = self.transforms[0](image=x.squeeze(), gt=y.squeeze()) 
                x = aug_data["image"]
                #aug_gt = self.transforms(image=y)
                y = aug_data["gt"]
                for i in range(1, len(self.transforms)):
                    x = self.transforms[i](image=x.squeeze())['image']
            
            else:
                #aug_data = self.transforms(image=x.squeeze()) #ikke noe problem med å legge til squeeze her hilsen Gabriel Kiss
                aug_data = self.transforms(image=x.squeeze(), gt=y.squeeze()) 
                x = aug_data["image"]
                #aug_gt = self.transforms(image=y)
                y = aug_data["gt"]
            x = np.expand_dims(x, 0)

        if self.model_depth:
            x = x.squeeze()
            img_shape = x.shape
            pad_x = 2**self.model_depth - img_shape[0] % 2**self.model_depth
            pad_y = 2**self.model_depth - img_shape[1] % 2**self.model_depth

            x = np.vstack([x, np.zeros((pad_x, x.shape[1]))])
            y = np.vstack([y, np.zeros((pad_x, x.shape[1]))])
            x = np.hstack([x, np.zeros((x.shape[0], pad_y))])
            y = np.hstack([y, np.zeros((y.shape[0], pad_y))])

            x = np.expand_dims(x, 0)

            #print('transformed x: ', x.shape)
            #print('trasformed y', y.shape)
        #print(np.amax(x))
        #print(np.amax(y))

        #print("x_mean", np.mean(x))
        #print("x_std", np.std(x))

        

        padding = [pad_x, pad_y]
        
        return x, y, padding, shape

    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    print(os.getcwd())
    from data.transforms import RandomCrop

    transtest = aug.Compose([
        #aug.augmentations.Resize(300, 300, interpolation=1, always_apply=False, p=1), #dette er for å resize bilde til ønsket størrelse
        #Resize(0, 0, fx=1, fy=1, interpolation=1, always_apply=False, p=1),
        #Padding(always_apply=False, p=1),
        #aug.augmentations.Resize(300, 300, interpolation=1, always_apply=False, p=1)
        #MAKE PADDIGN
        #RESIZE DOWN 
        #aug.augmentations.transforms.HorizontalFlip(p=1)
        #aug.augmentations.transforms.GaussianBlur(blur_limit=111, sigma_limit = 1, p=1) # Lagt til slik at ting kan blurres
        #aug.augmentations.transforms.Rotate(limit=90, p=0.5)
        #aug.augmentations.transforms.ElasticTransform(alpha=300, sigma=25, alpha_affine=1, interpolation=1, border_mode=1, always_apply=False, p=1)
        RandomCrop(1, 1)

    ], additional_targets={'gt': 'image',})
    test_trans = build_transforms(cfg, is_train=False, tee=False)

    

    dataset = DatasetLoader(cfg,
                            tee=True,
                            transforms=transtest)
    dataset.classes = [1, 2]
    
    # tee_data_loader = DataLoader(dataset,
    #                              num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    #                              pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    #                              batch_size=1,
    #                              #shuffle=True,
    #                              collate_fn)
    import matplotlib.pyplot as plt


    fig, axs = plt.subplots(2,4)
    ax =axs[0]
    ax2 = axs[1]
    #fig, ax2 = plt.subplots(1,5)
    ax2 = ax2.flat
    ax = ax.flat
    #random.seed(42)
    #for i in range(10):
    i = 0
    for data in Subset(dataset, range(0,4)):
        x = data[0]
        y = data[1]

        # print(gray.shape)
        # print(gt.shape)

        # for x, y in zip(gray, gt):
        values = []
        for j in y.flat:
            if not values.__contains__(j):
                values.append(j)
        print(values)
        
        ax[i].imshow(x.squeeze())
        ax2[i].imshow(y.squeeze())
        i += 1
    plt.legend()
    plt.show()
    plt.savefig('tee_rot.png')
        
    #     # os.exit()
