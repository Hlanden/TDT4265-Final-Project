import torch
from torch.utils.data import DataLoader
from data.datasets import build_dataset
from data.transforms import build_transforms, build_target_transform
from data.DatasetLoader import DatasetLoader
from pathlib import Path



# def make_data_loader(cfg, is_train=True, max_iter=None, start_iter=0):
#     train_transform = build_transforms(cfg, is_train=is_train)
#     target_transform = build_target_transform(cfg) if is_train else None
#     dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
#     datasets = build_dataset(
#         cfg.DATASET_DIR,
#         dataset_list, transform=train_transform,
#         target_transform=target_transform, is_train=is_train)

#     shuffle = is_train

#     data_loaders = []

#     for dataset in datasets:
#         if shuffle:
#             sampler = torch.utils.data.RandomSampler(dataset)
#         else:
#             sampler = torch.utils.data.sampler.SequentialSampler(dataset)

#         batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
#         batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=is_train)
#         if max_iter is not None:
#             batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter, start_iter=start_iter)

#         data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
#                                  pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train))
#         data_loaders.append(data_loader)

#     if is_train:
#         # during training, a single (possibly concatenated) data_loader is returned
#         assert len(data_loaders) == 1
#         return data_loaders[0]
#     return data_loaders

def make_data_loaders(cfg, is_train=True):
    train_transform = build_transforms(cfg, is_train=is_train)
    
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST


    data = DatasetLoader(Path(cfg.DATASETS.TRAIN_IMAGES), 
                        medimage=True,
                        transforms=transtest,
                        classes=[1])
    
    #base_path = Path('datasets/CAMUS_resized')
    #data = DatasetLoader(base_path + '\gray',
    #                     base_path + '\gt',
    #                    medimage=False,
    #                    #transforms=transtest,
    #                    classes=[1])
    #split the training dataset and initialize the data loaders
    train_dataset, valid_dataset = torch.utils.data.random_split(data, (1650, 150)) #TODO: Okay split? Ot more on valid?

    shuffle = is_train

    batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
    
    data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train))
    data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
