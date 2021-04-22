if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.getcwd())
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
import argparse
import albumentations as aug
from data.build import make_data_loaders

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


def main():

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
    
    depth = len(cfg.UNETSTRUCTURE.CONTRACTBLOCK)
    train_data_loader, valid_data_loader, test_data_loader = make_data_loaders(cfg, classes= cfg.MODEL.CLASSES, is_train=True, model_depth=depth)
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
   
    best_checkpoint_data = checkpointer.load(f=checkpointer.get_best_checkpoint_file(), use_latest = False)
    arguments.update(best_checkpoint_data)

    model.train(False)

    logger.info('Number of parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)/(1000000)))
    max_iter = cfg.SOLVER.MAX_ITER

    test_loss = 0
    acc = np.zeros((1, len(cfg.MODEL.CLASSES))).flatten()
    total_img = 0
    with torch.no_grad():
        for num_batches, (images, targets) in enumerate(test_data_loader):
            print('Image size: ', images[0].shape)
            batch_size = images.shape[0]
            total_img += batch_size
            images = torch_utils.to_cuda(images)
            targets = torch_utils.to_cuda(targets)
            outputs = model(images)
            test_loss += loss_fn(outputs, targets.long())*batch_size
            #acc += dice_score(outputs, targets) # TODO: Wait on working function
            test_dice_score = dice_score_multiclass(outputs, targets, len(cfg.MODEL.CLASSES),model).flatten()
            acc += test_dice_score*batch_size
        acc = acc/total_img
        test_loss = val_loss/total_img
        

        eval_result = {}
        for i, c in enumerate(cfg.MODEL.CLASSES): 
            eval_result['DICE Scores/Test - DICE Score, class {}'.format(c)] = acc[i]
        print("Final test loss", test_loss)        


    val_loss = 0
    acc = np.zeros((1, len(cfg.MODEL.CLASSES))).flatten()
    total_img = 0
    with torch.no_grad():
        for num_batches, (images, targets) in enumerate(test_data_loader):
            print('Image size: ', images[0].shape)
            batch_size = images.shape[0]
            total_img += batch_size
            images = torch_utils.to_cuda(images)
            targets = torch_utils.to_cuda(targets)
            outputs = model(images)
            val_loss += loss_fn(outputs, targets.long())*batch_size
            #acc += dice_score(outputs, targets) # TODO: Wait on working function
            val_dice_score = dice_score_multiclass(outputs, targets, len(cfg.MODEL.CLASSES),model).flatten()
            acc += val_dice_score*batch_size
        acc = acc/total_img
        val_loss = val_loss/total_img
        

        eval_result = {}
        for i, c in enumerate(cfg.MODEL.CLASSES): 
            eval_result['DICE Scores/Val - DICE Score, class {}'.format(c)] = acc[i]
        print("Final val loss", val_loss)   

    train_loss = 0
    acc = np.zeros((1, len(cfg.MODEL.CLASSES))).flatten()
    total_img = 0
    with torch.no_grad():
        for num_batches, (images, targets) in enumerate(test_data_loader):
            print('Image size: ', images[0].shape)
            batch_size = images.shape[0]
            total_img += batch_size
            images = torch_utils.to_cuda(images)
            targets = torch_utils.to_cuda(targets)
            outputs = model(images)
            train_loss += loss_fn(outputs, targets.long())*batch_size
            #acc += dice_score(outputs, targets) # TODO: Wait on working function
            train_dice_score = dice_score_multiclass(outputs, targets, len(cfg.MODEL.CLASSES),model).flatten()
            acc += train_dice_score*batch_size
        acc = acc/total_img
        train_loss = train_loss/total_img
        

        eval_result = {}
        for i, c in enumerate(cfg.MODEL.CLASSES): 
            eval_result['DICE Scores/Train - DICE Score, class {}'.format(c)] = acc[i]
        print("Final Train loss", Train loss)   

    

if __name__ == '__main__':
    main()