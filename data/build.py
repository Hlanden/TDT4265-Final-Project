from albumentations.augmentations import utils
from numpy.lib.arraypad import pad
from torch.utils.data import DataLoader, random_split, Subset
from data.transforms import build_transforms
from data.DatasetLoader import DatasetLoader
from pathlib import Path
from copy import copy
import torch
import numpy as np
from utils.torch_utils import to_cuda

def custom_collate(batch, tee=False):
    # Credits: https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    padding = np.array([item[2] for item in batch])

    max_x = 0
    max_y = 0
    for item in batch:
        max_x = item[0].shape[1] if item[0].shape[1] > max_x else max_x 
        max_y = item[0].shape[2] if item[0].shape[2] > max_y else max_y
    shapes = [item[3] for item in batch]
    org_targets = [item[4] for item in batch]   
    ## get sequence lengths
    #max_shape = max(shapes)
   
    images = np.zeros((len(data), 1, max_x, max_y))
    targets = np.zeros((len(data), max_x, max_y))
    #print('mx', max_x, 'my', max_y)
    #labels = torch.tensor(labels)

    for i in range(len(data)):
        j, k = data[i][0].shape
        
        pad_x = max_x - j if max_x - j > 0 else 0
        pad_y = max_y - k if max_y - k > 0 else 0
        padding[i, :] += [pad_x, pad_y]
        
        try:
            if pad_x:
                padded_img = np.vstack((data[i][0], np.zeros((pad_x, k))))
                padded_target = np.vstack((target[i], np.zeros((pad_x, k))))
            else:
                padded_img = data[i][0]
                padded_target = target[i]
            if pad_y:
                padded_target = np.hstack((padded_target, np.zeros((padded_img.shape[0], pad_y))))
                padded_img = np.hstack((padded_img, np.zeros((padded_img.shape[0], pad_y))))
        
        except Exception as e:
            print('max x', max_x)
            print('max y', max_y)
            #print('Padded: ', padded_img.shape[0])
            print('pad_y ', pad_y)

        images[i][0] = padded_img
        targets[i] = padded_target
        
    
    """
    Should return a tensor, but not possible when the sizes are different...
    https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14

    """
    images = torch.from_numpy(images)
    targets = torch.from_numpy(targets)
    return images.float(), targets.long(), shapes, padding, org_targets


def make_data_loaders(cfg,
                      tee=False):
    train_transform, only_img_transform = build_transforms(cfg, is_train=True, tee=tee)
    val_transform, only_img_val_transform = build_transforms(cfg, is_train=False, tee=tee)

    dataset = DatasetLoader(cfg, tee=tee)
    batch_size = cfg.TEST.BATCH_SIZE if not tee else cfg.TEST.BATCH_SIZE
    print('BATCH SIZE ', batch_size)
    torch.manual_seed(0)
    if not tee:
        test_dataset = Subset(dataset, range(1600,1800))
        train_and_val_dataset = Subset(dataset, range(0,1600))
        train_dataset, valid_dataset = random_split(train_and_val_dataset, (1200, 400))
        train_dataset.dataset = copy(dataset)


        train_dataset.dataset.transforms = [train_transform, only_img_transform]
        valid_dataset.dataset.transforms = [val_transform, only_img_val_transform] #val_transform
        test_dataset.dataset.transforms = [val_transform, only_img_val_transform] #val_transform

        train_data_loader = DataLoader(train_dataset,
                                       num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                       pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       collate_fn=custom_collate)
        val_data_loader = DataLoader(valid_dataset,
                                     num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                     pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=custom_collate)
        test_data_loader = DataLoader(test_dataset,
                                      num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                      batch_size=batch_size,
                                      #shuffle=True,
                                      collate_fn=custom_collate)
        return train_data_loader, val_data_loader, test_data_loader
    else:
        dataset.transforms = [val_transform, only_img_val_transform]
        tee_data_loader = DataLoader(dataset,
                                     num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                     pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                     batch_size=1,
                                     #shuffle=True,
                                     collate_fn=lambda x: custom_collate(x, tee=True))
        return tee_data_loader
