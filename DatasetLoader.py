import numpy as np
import torch
from torchvision import transforms, datasets

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, gray_dir, gt_dir, pytorch=True): #legger til muligheten for transform her
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, gt_dir) for f in gray_dir.iterdir() if not f.is_dir()]
        self.get_as_pil(1)
        self.pytorch = pytorch
        self.rot_deg = 180
        
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