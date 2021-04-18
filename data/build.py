from torch.utils.data import DataLoader, random_split
from data.transforms import build_transforms
from data.DatasetLoader import DatasetLoader
from pathlib import Path
from copy import copy

def make_data_loaders(cfg, classes=[1, 2], is_train=True):
    train_transform = build_transforms(cfg, is_train=True)
    val_transform = build_transforms(cfg, is_train=False)

    dataset_list = cfg.DATASETS.TRAIN_IMAGES if is_train else cfg.DATASETS.TEST_IMAGES

    dataset = DatasetLoader(Path(dataset_list),
                            medimage=True,
                            classes=classes)
    batch_size = cfg.TEST.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE

    train_dataset, valid_dataset, test_dataset = random_split(dataset, (1200, 400, 200))
    train_dataset.dataset = copy(dataset)
    
    valid_dataset.dataset.transforms = val_transform
    test_dataset.dataset.transforms = val_transform
    train_dataset.dataset.transforms = train_transform
    
    if is_train:
        train_data_loader = DataLoader(train_dataset,
                                       num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                       pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                       batch_size=batch_size,
                                       shuffle=True)

        val_data_loader = DataLoader(valid_dataset,
                                     num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                     pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                     batch_size=batch_size,
                                     shuffle=True)

        test_data_loader = DataLoader(test_dataset,
                                      num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                      batch_size=batch_size,
                                      shuffle=True)
        return train_data_loader, val_data_loader, test_dataset
    else:
        test_data_loader = DataLoader(test_dataset,
                                      num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                      batch_size=batch_size,
                                      shuffle=True)
        return test_data_loader
