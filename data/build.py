from torch.utils.data import DataLoader, random_split
from data.transforms import build_transforms
from data.DatasetLoader import DatasetLoader
from pathlib import Path

def make_data_loaders(cfg, classes=[1, 2], is_train=True):
    train_transform = build_transforms(cfg)

    dataset_list = cfg.DATASETS.TRAIN_IMAGES if is_train else cfg.DATASETS.TEST_IMAGES

    dataset = DatasetLoader(Path(dataset_list),
                            medimage=True,
                            transforms=train_transform if is_train else [],
                            classes=classes)
    batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE

    if is_train:
        num_of_images = dataset.__len__()
        num_of_train_images = int(num_of_images)*0.9
        num_of_val_images = num_of_images - num_of_train_images
        # TODO: Okay split? Ot more on valid?
        train_dataset, valid_dataset = random_split(dataset, (num_of_images, num_of_val_images))

        valid_dataset.transforms = []

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

        return train_data_loader, val_data_loader
    else:
        test_data_loader = DataLoader(dataset,
                                      num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                      batch_size=batch_size,
                                      shuffle=True)
        return test_data_loader
