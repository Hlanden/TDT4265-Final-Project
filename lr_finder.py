from tqdm import tqdm, trange
import math

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

from train import get_parser



class LearningRateFinder:
    """
    Train a model using different learning rates within a range to find the optimal learning rate.
    """

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 #device
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_history = {}
        self._model_init = model.state_dict()
        self._opt_init = optimizer.state_dict()
        #self.device = device

    def fit(self,
            data_loader,
            steps=100,
            min_lr=1e-10,
            max_lr=1,
            constant_increment=False
            ):
        """
        Trains the model for number of steps using varied learning rate and store the statistics
        """
        self.loss_history = {}
        self.model.train()
        current_lr = min_lr
        steps_counter = 0
        epochs = math.ceil(steps / len(data_loader))
        print(len(data_loader))
        steps_counter = 0

        progressbar = trange(epochs, desc='Progress')
        for epoch in progressbar:
            batch_iter = tqdm(enumerate(data_loader), 'Training', total=len(data_loader),
                              leave=False)

            #for iteration, (images, targets) in enumerate(data_loader, steps_counter):

            for i, (images, targets) in batch_iter:
              #  x, y = x.to(self.device), y.to(self.device)
                
                for param_group in self.optimizer.param_groups:
                    
                    param_group['lr'] = current_lr
                self.optimizer.zero_grad()
                
                out = self.model(images)
                
                
                loss = self.criterion(out, targets)
                loss.backward()
                self.optimizer.step()
                self.loss_history[current_lr] = loss.item()
                steps_counter += 1
                if steps_counter > steps:
                    steps_counter = 0
                    break

                if constant_increment:
                    current_lr += (max_lr - min_lr) / steps
                else:
                    current_lr = current_lr * (max_lr / min_lr) ** (1 / steps)

    def plot(self,
             smoothing=True,
             clipping=True,
             smoothing_factor=0.1
             ):
        """
        Shows loss vs learning rate(log scale) in a matplotlib plot
        """
        loss_data = pd.Series(list(self.loss_history.values()))
        lr_list = list(self.loss_history.keys())
        if smoothing:
            loss_data = loss_data.ewm(alpha=smoothing_factor).mean()
            loss_data = loss_data.divide(pd.Series(
                [1 - (1.0 - smoothing_factor) ** i for i in range(1, loss_data.shape[0] + 1)]))  # bias correction
        if clipping:
            loss_data = loss_data[10:-5]
            lr_list = lr_list[10:-5]
        plt.plot(lr_list, loss_data)
        plt.xscale('log')
        plt.title('Loss vs Learning rate')
        plt.xlabel('Learning rate (log scale)')
        plt.ylabel('Loss (exponential moving average)')
        plt.savefig("ploooot.png")
        plt.show()
        

    def reset(self):
        """
        Resets the model and optimizer to its initial state
        """
        self.model.load_state_dict(self._model_init)
        self.optimizer.load_state_dict(self._opt_init)
        print('Model and optimizer in initial state.')


def main():

    args = get_parser().parse_args()
    print(args)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    #output_dir = Path(cfg.OUTPUT_DIR)
    #output_dir.mkdir(exist_ok=True, parents=True)

    #logger = setup_logger("UNET", output_dir)
    #logger.info(args)

    #logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        #logger.info(config_str)
    #logger.info("Running with config:\n{}".format(cfg))
    
    depth = len(cfg.UNETSTRUCTURE.CONTRACTBLOCK)
    train_data_loader, valid_data_loader, test_data_loader = make_data_loaders(cfg, classes= cfg.MODEL.CLASSES, is_train=True, model_depth=depth)
     

    #logger = logging.getLogger('UNET.trainer')
    model = Unet2D(cfg)
    #model = torch_utils.to_cuda(model)      #TEMP

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    loss_fn = nn.CrossEntropyLoss()

    arguments = {"iteration": 0, "epoch": 0,"running_time": 0}
    #save_to_disk = True
    #checkpointer = CheckPointer(
    #    model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
     #   )
    #extra_checkpoint_data = checkpointer.load()
    #arguments.update(extra_checkpoint_data)
    #logger.info('Number of parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)/(1000000)))
    max_iter = cfg.SOLVER.MAX_ITER



    lr_finder = LearningRateFinder(model,loss_fn,optimizer)
    
    lr_finder.fit(train_data_loader,
            steps=100,
            min_lr=1e-7,
            max_lr=1,
            constant_increment=False
            )

    lr_finder.plot()
    lr_finder.reset()

if __name__ == "__main__":
    main()