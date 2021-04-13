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
from medimage import image

from torchvision.transforms.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as TF
import cv2

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, gray_dir, gt_dir,transforms=None, flip=False, pytorch=True): #legger til muligheten for transform her
        super().__init__()
        # TODO: Fix if statement below. Not obvious enough to understand what is happening!
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        #if gt_dir:
        #    self.files = [self.combine_files(f, gt_dir, False) for f in gray_dir.iterdir() if not f.is_dir()]
        #else:
        self.files = []
        for f in gray_dir.iterdir():
            filename, filetype = os.path.splitext(f)
            
            if not f.is_dir() and str(f).__contains__('.mhd') \
                and not filename.__contains__('gt') \
                and not filename.__contains__('sequence'): #TODO: What shall we do with sequence?
                    self.files.append(self.combine_files(filename, '', True))
        self.pytorch = pytorch
        #self.rot_deg = 180
        #self.flip = flip
        self.transforms = transforms
        self.medimage = medimage


    def combine_files(self, gray_file: Path, gt_dir, camus=True):
        if camus:
            files = {'gray': gray_file + '.mhd', 'gt': gray_file + '_gt.mhd'}
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
            raw_us = np.stack([raw_us.squeeze(), ], axis=2)
            #raw_us = np.expand_dims(raw_us, axis=0)
            print('Raw us shape', raw_us.shape)
            #raw_us = raw_us.squeeze()
            #print(raw_us.shape)
        else:
            raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                            ], axis=2)
    
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        #open mask file
        if self.medimage:
            raw_mask = image(self.files[idx]['gt']).imdata
            raw_mask = raw_mask.squeeze()
            gtbox_type = 3 # TODO: Change this to the correct type
            raw_mask_temp1 = raw_mask.copy()
            raw_mask_temp2 = raw_mask.copy()
            #raw_mask_temp3 = raw_mask
            raw_mask_temp1 = np.where(raw_mask==1, 1, 0)
            raw_mask_temp2 = np.where(raw_mask==2, 2, 0)
            raw_mask = (raw_mask_temp1 + raw_mask_temp2).copy()
        else:
            raw_mask = np.array(Image.open(self.files[idx]['gt']))            
            raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def rotate_image(self, img):
        return img.flip(2)
        

    def __getitem__(self, idx):
        #get the image and mask as arrays
<<<<<<< HEAD
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        print('y shape', y.shape)
=======
        #x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        #y = torch.tensor(self.open_mask(idx, add_dims=True), dtype=torch.torch.int64)

        x = self.open_as_array(idx, invert=self.pytorch).astype(np.float32)
        y = self.open_mask(idx, add_dims=False).astype(np.float32)

        if self.flip:
            x = self.rotate_image(x)
            y = self.rotate_image(y)
        if self.transforms:
                
            aug_data = self.transforms(image=x.squeeze()) #ikke noe problem med å legge til squeeze her hilsen Gabriel Kiss
            x = aug_data["image"]

            aug_data2 = self.transforms(image=y)
            y = aug_data2["image"]
        
>>>>>>> pre_pros_data
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
<<<<<<< HEAD
    
if __name__ == '__main__':
    # base_path = Path('data/CAMUS_resized')
    # data = DatasetLoader(base_path/'train_gray', 
    #                     base_path/'train_gt', 
    #                     medimage=False)

    # Simple test sxript for plotting of data + gt 
    
    data = DatasetLoader(Path('patient0001',''))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4) 
    i = 0

    ''
    for d in data:
        if i < 10:
            ax[0, i].imshow(d[0].squeeze())
            ax[1, i].imshow(d[1].squeeze())
            i += 1
        else:
            break
    plt.show()

=======


    # def rotate_all_images(self):
    #     for i in range(len(self.files)):
    #         gray_file_path = self.files[i]['gray'] #henter pathen til de forskjellige
    #         gt_file_path = self.files[i]['gt']

    #         rotate_img(gray_file_path, self.rot_deg)
    #         rotate_img(gt_file_path, self.rot_deg)
        



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    transtest = aug.Compose([
        aug.augmentations.Resize(300, 300, interpolation=1, always_apply=False, p=1), #dette er for å resize bilde til ønsket størrelse
        #aug.HorizontalFlip(p=1)
    ])


    base_path = Path('data/CAMUS_resized')
    datasetloader = DatasetLoader(base_path/'train_gray', 
                        base_path/'train_gt',
                        transforms=transtest)

    train_data = DataLoader(datasetloader, batch_size=5, shuffle=True)
    fig, axs = plt.subplots(2,5)
    ax =axs[0]
    ax2 = axs[1]
    #fig, ax2 = plt.subplots(1,5)
    ax2 = ax2.flat
    ax = ax.flat
    for data in train_data:
        i = 0
        gray = data[0]
        gt = data[1]

        for x, y in zip(gray, gt):

            ax[i].imshow(x.squeeze())
            ax2[i].imshow(y.squeeze())
            i += 1
        plt.show()
        print(i)
        os.exit()


    fig, ax = plt.subplots(2,5)
    fig, ax2 = plt.subplots(2,5)
    ax2 = ax2.flat
    ax = ax.flat
    for i in range(10):
        x, y = datasetloader[i]
        x = np.squeeze(x[0])
        y = np.squeeze(y[0])
        #im = Image.open(x)
        ax[i].imshow(x)
        ax2[i].imshow(y)
    plt.show()


>>>>>>> pre_pros_data
