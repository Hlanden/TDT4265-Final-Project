import numpy as np
from numpy.core import numeric
import torch
from torchvision import transforms, datasets

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, dataloader, sampler
from PIL import Image
from torchvision.transforms.transforms import ToPILImage, ToTensor

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, gray_dir, gt_dir,transforms=None, pytorch=True): #legger til muligheten for transform her
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, gt_dir) for f in gray_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        self.rot_deg = 180
        self.transforms = transforms
        
    def combine_files(self, gray_file: Path, gt_dir):
        
        files = {'gray': gray_file, 
                 'gt': gt_dir/gray_file.name.replace('gray', 'gt')}

        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                           ], axis=2)
    
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        #get the image and mask as arrays
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        #x = self.open_as_array(idx, invert=self.pytorch)
        #y = self.open_mask(idx, add_dims=False)
        sample = {'gray': x, 'gt': y}
        if self.transforms:
            try:
                x = self.transforms(x)
                y = self.transforms(y)
            #Quick fix for random rotation
            # TODO: Fix this error (4D 5D)
            except RuntimeError as E:
                print('ERRRROR')
                x = self.transforms(x.unsqueeze(0))
                y = self.transforms(y.unsqueeze(0))
        
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')

    # def convertToIsoTropicPixelSize(self):
    #     data_transform = transforms.Compose([
    #                 transforms.RandomSizedCrop(224)
    #     ])

    #     self.files
    def add_rotation_transform(self):
        self.transforms

    def rotate_all_images(self):
        for i in range(len(self.files)):
            gray_file_path = self.files[i]['gray'] #henter pathen til de forskjellige
            gt_file_path = self.files[i]['gt']

            rotate_img(gray_file_path, self.rot_deg)
            rotate_img(gt_file_path, self.rot_deg)
        
def rotate_img(img_path, deg): #her må jeg få tak i riktig img path også også erstatte originalt bilde med dette
        img = Image.open(img_path)
        img.rotate(rt_degr, expand=1)
        img.save(path) # to override your old file


    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    trans = transforms.Compose([
        transforms.RandomCrop(150),
        transforms.RandomRotation(30),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.ToTensor()
    ]
    )
    base_path = Path('data/CAMUS_resized')
    datasetloader = DatasetLoader(base_path/'train_gray', 
                        base_path/'train_gt',
                        transforms=trans)

    train_data = DataLoader(datasetloader, batch_size=5, shuffle=True)
    fig, ax = plt.subplots(1,5)
    fig, ax2 = plt.subplots(1,5)
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

    # dataloader = DataLoader(datasetloader, batch_size=4,
    #                     shuffle=True, num_workers=0)
    
    # i = 0
    # for x, y in dataloader:
    #     print(x)
    #     print(y)
    #     fig, ax = plt.subplots(1,2)
    #     ax[0].imshow(x[0])
    #     ax[1].imshow(y[0])
    #     plt.show()
    #     i += 1
    #     input('Press enter for next')
    #     if i > 10:
    #         os.exit(-1)


