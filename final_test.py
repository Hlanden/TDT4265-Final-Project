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
from backboned_unet.unet import Unet 


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

from utils.evaluation import dice_score_multiclass
from tqdm import tqdm

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
    
    train_data_loader, val_data_loader, test_data_loader = make_data_loaders(cfg)
    tee_data_loader = make_data_loaders(cfg, tee=True)

    loaders = {'Train': train_data_loader,
               'Validation': val_data_loader,
               'Test': test_data_loader,
               'TEE': tee_data_loader}



    logger = logging.getLogger('UNET.trainer')
    if cfg.MODEL.BACKBONE.USE:
        model = Unet(
                backbone_name= cfg.MODEL.BACKBONE.NET,
                pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                encoder_freeze=cfg.MODEL.BACKBONE.ENCODER_FREZE,
                classes = cfg.MODEL.OUT_CHANNELS,
                decoder_filters=cfg.MODEL.BACKBONE.DECODER_FILTERS ,
                parametric_upsampling=cfg.MODEL.BACKBONE.PARAMETRIC_UPSAMPLING ,
                shortcut_features=cfg.MODEL.BACKBONE.SHORTCUT_FEATURES,
                decoder_use_batchnorm=cfg.MODEL.BACKBONE.DECODER_USE_BATCHNORM,)

    else:
        model = Unet2D(cfg)
    model = torch_utils.to_cuda(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    loss_fn = nn.CrossEntropyLoss()

    arguments = {"iteration": 0, "epoch": 0,"running_time": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
        )
   
    best_checkpoint_data = checkpointer.load(f=checkpointer.get_best_checkpoint_file(),
                                             use_latest=False)
    arguments.update(best_checkpoint_data)

    model.train(False)

    logger.info('Number of parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)/(1000000)))

    results = {}
    
    for loader_name, loader in loaders.items():
        logger.info('Running test on {}'.format(loader_name))
        results[loader_name] = []
        loss = 0
        acc = np.zeros((1, len(cfg.MODEL.CLASSES))).flatten()
        total_img = 0
        with torch.no_grad():
            for num_batches, (images, targets, shapes, padding) in enumerate(tqdm(loader)):
                batch_size = images.shape[0]
                total_img += batch_size
                images = torch_utils.to_cuda(images)
                targets = torch_utils.to_cuda(targets)
                outputs = model(images)
                loss += loss_fn(outputs, targets.long())*batch_size
                dice_score = dice_score_multiclass(outputs, targets, len(cfg.MODEL.CLASSES),shapes=shapes, padding=padding).flatten()
                acc += dice_score*batch_size
            acc = acc/total_img
            loss = loss/total_img
            
            results[loader_name].append(loss)
            eval_result = {}
            for i, c in enumerate(cfg.MODEL.CLASSES): 
                eval_result['DICE Scores/Train - DICE Score, class {}'.format(c)] = acc[i]
                results[loader_name].append(acc[i])
            logger.info('{} result: {},\n{} loss: {}'.format(loader_name, eval_result, loader_name, loss))  

if __name__ == '__main__':
    main()
