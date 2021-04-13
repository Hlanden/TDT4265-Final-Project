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

from DatasetLoader import DatasetLoader
from Unet2D import Unet2D

from config.defaults import cfg
from utils.logger import setup_logger
import utils.torch_utils as torch_utils
from utils.checkpoint import CheckPointer 
from engine.trainer import do_train
import argparse
import albumentations as aug

def start_train(cfg, train_loader):
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
    logger = logging.getLogger('SSD.trainer')
    model = Unet2D(cfg)
    model = torch_utils.to_cuda(model)

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=cfg.SOLVER.LR,
    #     momentum=cfg.SOLVER.MOMENTUM,
    #     weight_decay=cfg.SOLVER.WEIGHT_DECAY
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    loss_fn = nn.CrossEntropyLoss()

    arguments = {"iteration": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
        )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER
    # TODO: Import dataloader here
    #train_loader = make_data_loader(cfg, is_train=True, max_iter=max_iter, start_iter=arguments['iteration'])

    model = do_train(
        cfg, model, train_loader, optimizer,
        checkpointer, arguments, loss_fn)
    return model



def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  dice/acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_score(predb, yb):
    predb = predb.argmax(dim=1)
    predb = predb.view(-1)
    yb = yb.view(-1)
    intersection = (predb * yb).sum()                      
    dice = (2*intersection)/(predb.sum() + yb.sum())
    return dice

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

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
    

    #enable if you want to see some plotting
    visual_debug = cfg.LOGGER.VISUAL_DEBUG

    batch_size = cfg.TEST.BATCH_SIZE
    number_of_epochs = cfg.TEST.NUM_EPOCHS
    learning_rate = cfg.SOLVER.LR

    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)


    logger = setup_logger("UNET", output_dir)
    logger.info(args)
    #sets the matplotlib display backend (most likely not needed)
    #mp.use('TkAgg', force=True)                    #COMMENTED OUT IN ORDER TO RUN THE CODE. LUDVIK
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))


    # TODO: Move this out of main
    #load the training data
    transtest = aug.Compose([
        aug.augmentations.Resize(384, 384, interpolation=1, always_apply=False, p=1) 
    ])
    
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
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for i, (inputs, targets) in enumerate(train_data):
        with torch.no_grad():
                print("targets.data", inputs.shape)
                print('Targets', targets.shape)
                break
    valid_data = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    start_train(cfg, train_data)
    if visual_debug:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data.open_as_array(150))
        ax[1].imshow(data.open_mask(150))
        logger.info('Showing visual plotting')
        plt.show()

    #xb, yb = next(iter(train_data))
    #print (xb.shape, yb.shape)

    ## build the Unet2D with one channel as input and 2 channels as output
    #unet = Unet2D(cfg)

    ##loss function and optimizer
    #loss_fn = nn.CrossEntropyLoss()

    #opt = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    ##do some training
    #train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, dice_score, epochs=number_of_epochs)

    ##plot training and validation losses
    #if visual_debug:
    #    plt.figure(figsize=(10,8))
    #    plt.plot(train_loss, label='Train loss')
    #    plt.plot(valid_loss, label='Valid loss')
    #    plt.legend()
    #    plt.show()

    ##predict on the next train batch (is this fair?)
    #xb, yb = next(iter(train_data))
    #with torch.no_grad():
    #    predb = unet(xb.cuda())

    ##show the predicted segmentations
    #if visual_debug:
    #    fig, ax = plt.subplots(batch_size,3, figsize=(15,batch_size*5))
    #    for i in range(batch_size):
    #        ax[i,0].imshow(batch_to_img(xb,i))
    #        ax[i,1].imshow(yb[i])
    #        ax[i,2].imshow(predb_to_mask(predb, i))

    #    plt.show()

if __name__ == "__main__":
    main()
