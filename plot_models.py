import matplotlib.pyplot as plt
from torch.autograd.grad_mode import F
from data.build import make_data_loaders
from config.defaults import cfg
from pathlib import Path
from Unet2D import Unet2D
import utils.torch_utils as torch_utils
import torch
from utils.checkpoint import CheckPointer
import logging
from utils.logger import setup_logger
from data.transforms import build_transforms
import numpy as np
import argparse
from data.DatasetLoader import DatasetLoader

def predb_to_mask(predb, idx):
    #p = torch.functional.F.softmax(predb[idx], 0)
    return predb.argmax(dim=1).cpu()


def plot_model_from_checkpoint(cfg,
                               dataset_loader,
                               checkpoint,
                               image_idx=range(1, 10),
                               axs=None,
                               config_file=False,
                               filename=''):
    if config_file:
        cfg.merge_from_file(config_file)
    cfg.freeze()
    cp_dir = Path(cfg.OUTPUT_DIR)
    cp_path = ''
    for cp in cp_dir.iterdir():
        if str(cp).__contains__('pth') and \
            str(cp)[-4-len(str(checkpoint)):-4].__contains__(str(checkpoint)) and \
                str(cp)[-5-len(str(checkpoint))] == '0':
            cp_path = cp
            break
    if not cp_path:
        print('Invalid checkpoint')
        return

    model = Unet2D(cfg)
    model = torch_utils.to_cuda(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    logger = setup_logger('Plotter')
    save_to_disk = False

    checkpointer = CheckPointer(model,
                                optimizer,
                                cfg.OUTPUT_DIR,
                                save_to_disk,
                                logger,
                                )
    extra_checkpoint_data = checkpointer.load(f=str(cp_path), use_latest=False)

    if axs is None:
        fig, axs = plt.subplots(len(image_idx), 3)
        fig.suptitle('Checkpoint {}'.format(checkpoint))
        axs[0, 0].set_title('Original image')
        axs[0, 1].set_title('Ground thruths')
        axs[0, 2].set_title('Model images')
    
    for idx, ax in zip(image_idx, axs):
        image, target = dataset_loader[idx]
        output = model(torch.Tensor(np.expand_dims(image,0)).cuda())
        output = output.argmax(dim=1).cpu()
        ax[0].imshow(image.squeeze())
        ax[1].imshow(target.squeeze())
        ax[2].imshow(output.squeeze())
        

    if filename:
        print('Saving to file: {}'.format(filename))
        plt.savefig(filename + '.png')
    

def plot_mulitple_chekpoints(cfg,
                             dataset_loader,
                             checkpoints,
                             image_idx,
                             config_file=False, 
                             filename=''):
    if config_file:
        cfg.merge_from_file(config_file)
    cfg.freeze()
    fig, axs = plt.subplots(len(checkpoints), 3, figsize=(15, 8))
    #fig.tight_layout(pad=1.5)
    fig.suptitle('Results checkpoint: {}'.format(checkpoints))
    is_titles_set = False

    for i in range(len(checkpoints)):
        cp = checkpoints[i]
        ax = axs[i]
        if not is_titles_set:
            ax[0].set_title('Original image')
            ax[1].set_title('Ground thruths')
            ax[2].set_title('Model images')
            is_titles_set = True
        #ax[0].set_ylabel(str(cp), size='large', rotation=0)
        pad=5
        ax[0].annotate(str(cp), xy=(0, 0.5), xytext=(-ax[0].yaxis.labelpad - pad, 0),
                xycoords=ax[0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
        plot_model_from_checkpoint(cfg,
                                   dataset_loader,
                                   cp,
                                   image_idx=[image_idx],
                                   axs=[ax],
                                   filename='')

    if filename:
        print('Saving to file: {}'.format(filename))
        plt.savefig(filename + '.png')



if __name__ == '__main__':
    train_data_loader, valid_data_loader, _ = make_data_loaders(cfg, classes=cfg.MODEL.CLASSES, is_train=True)
    transforms = build_transforms(cfg, is_plotting=True)
    dataset_list = cfg.DATASETS.TRAIN_IMAGES
    dataset = DatasetLoader(Path(dataset_list),
                            medimage=True,
                            classes=cfg.MODEL.CLASSES, 
                            transforms=transforms)
    plot_mulitple_chekpoints(cfg, dataset, [500, 1000, 1500], 150, config_file='config/models/CAMUS.yaml', filename='test')
