import os
print(os.getcwd())
import matplotlib.pyplot as plt
from data.build import make_data_loaders
from config.defaults import cfg
from pathlib import Path
from Unet2D import Unet2D
import utils.torch_utils as torch_utils
import torch
from utils.checkpoint import CheckPointer
import logging
from utils.logger import setup_logger
import numpy as np

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def plot_model_from_checkpoint(cfg, dataloader, checkpoint, image_idx=range(1, 10), axs=None):
    cp_dir = Path(cfg.OUTPUT_DIR)
    cp_path = ''
    for cp in cp_dir.iterdir():
        print('Checkpoint: ', cp[-len(checkpoint):-4])
        if cp[-len(checkpoint):-4].__contains__(str(checkpoint)):
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
    if axs is None:
        fig, axs = plt.subplots(len(image_idx), 3)
    for idx, ax in zip(image_idx, axs):
        image, target = dataloader[idx]
        output = model(torch.tensor(np.expand_dims(image,0)).cuda())
        ax[idx,0].imshow(image.squeeze())
        ax[idx,1].imshow(target.squeeze())
        ax[idx,2].imshow(predb_to_mask(output,0))

def plot_mulitple_chekpoints(cfg, dataloader, checkpoints, image_idx):
    fig, axs = plt.subplots(len(checkpoints), 3)
    for cp, ax in zip(checkpoints, axs):
        plot_model_from_checkpoint(cfg,
                                   dataloader,
                                   cp,
                                   image_idx=[image_idx],
                                   axs=[ax])


if __name__ == '__main__':
    train_data_loader, valid_data_loader = make_data_loaders(cfg, classes=cfg.MODEL.CLASSES, is_train=True)
    plot_mulitple_chekpoints(cfg, train_data_loader, [500, 1000, 1500])
