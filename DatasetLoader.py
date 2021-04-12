from matplotlib.colors import cnames
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import os
from medimage import image

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, gray_dir, gt_dir='', medimage=True, pytorch=True, classes=[1,2]):
        super().__init__()
        # TODO: Fix if statement below. Not obvious enough to understand what is happening!
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        if gt_dir:
            self.files = [self.combine_files(f, gt_dir, False) for f in gray_dir.iterdir() if not f.is_dir()]
        else:
            self.files = []
            for f in gray_dir.iterdir():
                filename, filetype = os.path.splitext(f)
                
                if not f.is_dir() and str(f).__contains__('.mhd') \
                    and not filename.__contains__('gt') \
                    and not filename.__contains__('sequence'): #TODO: What shall we do with sequence?
                        self.files.append(self.combine_files(filename, '', True))
        self.pytorch = pytorch
        self.medimage = medimage
        self.classes = classes

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
            combined_classes = np.zeros_like(raw_mask)
            for c in self.classes:
                combined_classes += np.where(raw_mask==c, c, 0).astype(np.uint8)
            raw_mask = combined_classes
        else:
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
    for d in data:
        if i < 10:
            ax[0, i].imshow(d[0].squeeze())
            ax[1, i].imshow(d[1].squeeze())
            i += 1
        else:
            break
    plt.show()

