from matplotlib.colors import cnames
import numpy as np
from numpy.core import numeric
import torch
from torchvision import transforms, datasets
import albumentations as aug 

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, dataloader, sampler
from PIL import Image
import os
from medimage.medimage import image
from torchvision.transforms.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as TF
import cv2

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self,
                 gray_dir,
                 gt_dir='',
                 transforms=None,
                 flip=False,
                 pytorch=True,
                 medimage=True,
                 classes=[1, 2]):  # legger til muligheten for transform her

        super().__init__()
        # TODO: Fix if statement below. Not obvious enough to understand what is happening!
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        if gt_dir:
            self.files = [self.combine_files(f, gt_dir, False) for f in gray_dir.iterdir() if not f.is_dir()]
        else:
            self.files = []
            for patient in gray_dir.iterdir():
                for f in patient.iterdir():
                    filename, filetype = os.path.splitext(f)

                    if not f.is_dir() and str(f).__contains__('.mhd') \
                        and not filename.__contains__('gt') \
                        and not filename.__contains__('sequence'): #TODO: What shall we do with sequence?
                            self.files.append(self.combine_files(filename, '', True))
        self.pytorch = pytorch
        self.medimage = medimage
        self.classes = classes
        self.flip = flip
        self.transforms = transforms

    def combine_files(self, gray_file: Path, gt_dir, camus=True):
        if camus:
            files = {'gray': gray_file + '.mhd', 
                    'gt': gray_file + '_gt.mhd'}

        else:
            files = {'gray': gray_file, 
                    'gt': gt_dir/gray_file.name.replace('gray', 'gt')}

        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        if self.medimage:
            raw_us = image(self.files[idx]['gray']).imdata
        else:
            raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                              ], axis=2) #exkra dim legges til for å ha kontroll på batch size

    
        if invert:
            raw_us = raw_us.transpose((2,0,1))

        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False, transforms=False):
        #open mask file
        if self.medimage:
            raw_mask = image(self.files[idx]['gt']).imdata
            raw_mask = raw_mask.squeeze()
            combined_classes = np.zeros_like(raw_mask)
            for c in self.classes:
                combined_classes += np.where(raw_mask==c, c, 0).astype(np.uint8)
            raw_mask = combined_classes

        else:
            raw_mask = np.array(Image.open(self.files[idx]['gt']))            
            raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def rotate_image(self, img):
        return img.flip(2)
        

    def __getitem__(self, idx):
        #get the image and mask as arrays
        #x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        #y = torch.tensor(self.open_mask(idx, add_dims=True), dtype=torch.torch.int64)

        x = self.open_as_array(idx, invert=self.pytorch).astype(np.float32)
        y = self.open_mask(idx, add_dims=False).astype(np.float32)

        # if self.flip:
        #     x = self.rotate_image(x)
        #     y = self.rotate_image(y)
        if self.transforms:
                
            #aug_data = self.transforms(image=x.squeeze()) #ikke noe problem med å legge til squeeze her hilsen Gabriel Kiss
            aug_gray = self.transforms(image=x.squeeze()) 
            x = np.expand_dims(aug_gray["image"], 0)
            

            aug_gt = self.transforms(image=y)
            y = aug_gt["image"]
        
        
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    pjap = False
    skap = True
    rap = False
    transtest = aug.Compose([
        #aug.augmentations.Resize(300, 300, interpolation=1, always_apply=False, p=1), #dette er for å resize bilde til ønsket størrelse
        
        #aug.augmentations.transforms.HorizontalFlip(p=1)
        #aug.augmentations.transforms.GaussianBlur(blur_limit=111, sigma_limit = 0, p=1) # Lagt til slik at ting kan blurres

        #Denne slet jeg litt med å få immplementert
        #aug.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)


        #aug.augmentations.transforms.RandomCrop (300, 300, always_apply=False, p=1.0),

        #Her må jeg passe på at de blir rotert likt
        aug.augmentations.Rotate(limit=90, interpolation=1, border_mode=cv2.BORDER_WRAP, value=None, mask_value=None, always_apply=False, p=1)

    ])
    
    data = DatasetLoader(Path('patients',''),gt_dir='' , transforms=transtest)

    train_data = DataLoader(data, batch_size=5, shuffle=True)
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2,4)
    ax =axs[0]
    ax2 = axs[1]
    #fig, ax2 = plt.subplots(1,5)
    ax2 = ax2.flat
    ax = ax.flat
    for data in train_data:
        i = 0
        gray = data[0]
        gt = data[1]

        #print(gray.shape)
        #print(gt.shape)

        for x, y in zip(gray, gt):

            print(x.shape)
            print(y.shape)

            ax[i].imshow(x.squeeze())
            ax2[i].imshow(y.squeeze())
            i += 1
        plt.show()
        print(i)
        os.exit()
