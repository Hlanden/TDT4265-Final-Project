import argparse
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time
import logging

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

from data.DatasetLoader import DatasetLoader
from Unet2D import Unet2D

from config.defaults import cfg
from utils.logger import setup_logger
import utils.torch_utils as torch_utils
from utils.checkpoint import CheckPointer 
from engine.trainer import do_train
import argparse
import albumentations as aug
from data.build import make_data_loaders

def start_train(cfg, train_data_loader, val_data_loader):
    """
    Starts training the model with configurations defined by cfg.

    TODO: Fill in more detailed description here. 

    Input arguments: 
        cfg [yacs.configCfgNode] -- Model configuration 
        trainloader[pytorch.Dataloader] -- Dataloader for training set. TODO: Remove this
            should be configured in the cfg file, not explicit
    
    Returns: 
        model [torch.nn.Module] -- Trained model
    
    """
    logger = logging.getLogger('UNET.trainer')
    model = Unet2D(cfg)
    model = torch_utils.to_cuda(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    loss_fn = nn.CrossEntropyLoss()

    arguments = {"iteration": 0, "epoch": 0,"running_time": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
        )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER

    model = do_train(
        cfg, model, train_data_loader, val_data_loader, optimizer,
        checkpointer, arguments, loss_fn)
    return model


def get_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config_file",
        default="config/models/CAMUS.yaml",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def main ():
    args = get_parser().parse_args()
    print(args)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger("UNET", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train_data_loader, valid_data_loader, test_data_loader = make_data_loaders(cfg, classes= cfg.MODEL.CLASSES, is_train=True)

    model = start_train(cfg, train_data_loader, valid_data_loader)
    
if __name__ == "__main__":
    main()
