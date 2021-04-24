import argparse
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time
import logging
#import sys

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



from backboned_unet.unet import Unet 
from backboned_unet.utils import DiceLoss

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


    #maybe go inn and tweak here???
    if cfg.MODEL.BACKBONE.USE:
        model = Unet(
                backbone_name= cfg.MODEL.BACKBONE.NET,
                pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                encoder_freeze=cfg.MODEL.BACKBONE.ENCODER_FREZE,
                classes = cfg.MODEL.OUT_CHANNELS,
                decoder_filters=cfg.MODEL.BACKBONE.DECODER_FILTERS ,
                parametric_upsampling=cfg.MODEL.BACKBONE.PARAMETRIC_UPSAMPLING ,
                shortcut_features=cfg.MODEL.BACKBONE.SHORTCUT_FEATURES,
                decoder_use_batchnorm=cfg.MODEL.BACKBONE.DECODER_USE_BATCHNORM,
                cfg=cfg)

    else:
        model = Unet2D(cfg)
    model = torch_utils.to_cuda(model)

    if cfg.SOLVER.DIFFRENT:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.LR)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    if cfg.LOSS.DIFFRENT:
        loss_fn = nn.DiscLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    arguments = {"iteration": 0, "epoch": 0,"running_time": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
        )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)
    logger.info('Number of parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)/(1000000)))
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

def main (logger=None):
    args = get_parser().parse_args()
    print(args)
    print(cfg.PREPROCESSING.RANDOMCROP.ENABLE)
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
    
    depth = len(cfg.UNETSTRUCTURE.CONTRACTBLOCK)
    train_data_loader, valid_data_loader, test_data_loader = make_data_loaders(cfg)
     

    model = start_train(cfg, train_data_loader, valid_data_loader)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

def load_best_model(cfg): #Can we delete this?
    logger = logging.getLogger('UNET.test')
    model = Unet2D(cfg)
    model = torch_utils.to_cuda(model)

    if cfg.SOLVER.DIFFRENT:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.LR)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)


    if cfg.LOSS.DIFFRENT:
        loss_fn = nn.DiceLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    arguments = {"iteration": 0, "epoch": 0,"running_time": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
        )
    
    with open(cfg.OUTPUT_DIR + '/best_checkpoint.txt') as best_file:
        f = best_file.read().strip()
    extra_checkpoint_data = checkpointer.load(f=f,use_latest=False)
    arguments.update(extra_checkpoint_data)
    return model


if __name__ == "__main__":
    #import sys
    #sys.argv.append('--config_file=config/models/backbone_resnet34.yaml')
    main()
    #sys.argv[1] = '--config_file=config/models/backbone_vgg16.yaml'
    #main()
    #sys.argv[1] = '--config_file=config/models/backbone_resnet34.yaml'
    #main()

