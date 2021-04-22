from torch.utils.data import DataLoader, random_split, Subset
from data.transforms import build_transforms
from data.DatasetLoader import DatasetLoader
from pathlib import Path
from copy import copy
import torch

def custom_collate(batch):
    # Credits: https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    shapes = [item[0][0].shape for item in batch]
    ## get sequence lengths
    max_len = max(shapes)
    features = torch.zeros((len(data), *max_len))
    print('Festures size: ', features.shape)
    #labels = torch.tensor(labels)
    shapes = torch.tensor(shapes)

    for i in range(len(data)):
        j, k = data[i][0].shape
        print('Original shape: ', data[i][0].shape)
        print('Cat x: ', max_len[0]-j)
        print('Cat y:', max_len[1]-k)
        print('Final shape: ', data[i][0].shape[0]+max_len[0]-j, data[i][0].shape[1]+max_len[1]-k)
        print('Max shape: ', max_len)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len[0]-j, k))])
        print('Final 2: ', features[i].shape)
    
    """
    Should return a tensor, but not possible when the sizes are different...
    https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14

    """
    return features.float(), targets.long(), shapes.long()

def make_data_loaders(cfg, classes=[1, 2], is_train=True, model_depth=False):
    train_transform = build_transforms(cfg, is_train=True)
    val_transform = build_transforms(cfg, is_train=False)

    dataset_list = cfg.DATASETS.TRAIN_IMAGES if is_train else cfg.DATASETS.TEST_IMAGES

    dataset = DatasetLoader(Path(dataset_list),
                            medimage=True,
                            classes=classes,
                            model_depth=model_depth)
    batch_size = cfg.TEST.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE

    test_dataset = Subset(dataset, range(1600,1800))
    train_val_dataset = Subset(dataset, range(0,1600))
    train_dataset, valid_dataset = random_split(train_val_dataset, (1200, 400))
    train_dataset.dataset = copy(dataset)
    
    valid_dataset.dataset.transforms = val_transform
    test_dataset.dataset.transforms = val_transform
    train_dataset.dataset.transforms = train_transform
    
    if is_train:
        train_data_loader = DataLoader(train_dataset,
                                       num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                       pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                       batch_size=batch_size,
                                       shuffle=True,)

        val_data_loader = DataLoader(valid_dataset,
                                     num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                     pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                     batch_size=batch_size,
                                     shuffle=True,)

        test_data_loader = DataLoader(test_dataset,
                                      num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                      batch_size=batch_size,
                                      shuffle=True,)
        return train_data_loader, val_data_loader, test_dataset
    else:
        test_data_loader = DataLoader(test_dataset,
                                      num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=custom_collate)
        return test_data_loader
