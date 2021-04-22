import collections
import datetime
import logging
import os
import time
import torch
from torch.utils.data import dataloader
import numpy as np
import torch.utils.tensorboard
from engine.inference import do_evaluation
from utils.metric_logger import MetricLogger
from utils import torch_utils
from utils.evaluation import dice_score_multiclass
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)

def do_train(cfg, model,
             train_data_loader,
             val_data_loader,
             optimizer,
             checkpointer,
             arguments,
             loss_fn):
    logger = logging.getLogger("UNET.trainer")
    logger.info(input('Hvorfor tester du dette? '))
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()
    
    lr_finder = LRFinder(model, optimizer, loss_fn, device="cuda")

    summary_writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))

    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    lowest_loss = 1
    early_stopping_count = 0
    is_early_stopping = False
    epoch = arguments["epoch"]
    while (time.time() - start_training_time)/60 <= cfg.SOLVER.MAX_MINUTES and not is_early_stopping:
        epoch += 1
        arguments["epoch"] = epoch
        
        
        for iteration, (images, targets) in enumerate(train_data_loader, start_iter):
            iteration = iteration + 1
            arguments["iteration"] = iteration
            images = torch_utils.to_cuda(images).float()
            targets = torch_utils.to_cuda(targets)
            outputs = model(images)
            
            loss = loss_fn(outputs, targets.long())

            meters.update(total_loss=loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            arguments["running_time"] += batch_time
            end = time.time()
            meters.update(time=batch_time)
            if iteration % cfg.LOG_STEP == 0:
                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                to_log = [
                    f"iter: {iteration:06d}",
                    f"lr: {lr:.5f}",
                    f'{meters}',
                    f"eta: {eta_string}",
                ]
                if torch.cuda.is_available():
                    mem = round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                    to_log.append(f'mem: {mem}M')
                logger.info(meters.delimiter.join(to_log))
                global_step = iteration
                summary_writer.add_scalar(
                    'losses/train_loss', loss, global_step=global_step)
                
                summary_writer.add_scalar(
                    'lr', optimizer.param_groups[0]['lr'],
                    global_step=global_step)

                train_acc = dice_score_multiclass(outputs, targets, len(cfg.MODEL.CLASSES),model).flatten()
                train_acc_result = {}
                for i, c in enumerate(cfg.MODEL.CLASSES): 
                    train_acc_result['DICE Scores/Train - DICE Score, class {}'.format(c)] = train_acc[i]
                for key, acc in train_acc_result.items():
                    summary_writer.add_scalar(key, acc, global_step=global_step)
            
            if iteration >= cfg.SOLVER.MAX_ITER or is_early_stopping:
                break
         # TODO: Currently deactivated. Need dataloader class to make eval
        if cfg.EVAL_AND_SAVE_EPOCH > 0 and epoch % cfg.EVAL_AND_SAVE_EPOCH == 0 and epoch > 0:
            logger.info('Evaluating...')
            model.train(False)
            acc = np.zeros((1, len(cfg.MODEL.CLASSES))).flatten()
            val_loss = 0
            with torch.no_grad():
                for num_batches, (images, targets) in enumerate(val_data_loader):
                    images = torch_utils.to_cuda(images)
                    targets = torch_utils.to_cuda(targets)
                    outputs = model(images)
                    val_loss += loss_fn(outputs, targets.long())
                    #acc += dice_score(outputs, targets) # TODO: Wait on working function
                    val_dice_score = dice_score_multiclass(outputs, targets, len(cfg.MODEL.CLASSES),model).flatten()
                    acc += val_dice_score
                acc = acc/(num_batches+1)
                val_loss = val_loss/(num_batches+1)

                # Tensorboard logging
                eval_result = {}
                for i, c in enumerate(cfg.MODEL.CLASSES): 
                    eval_result['DICE Scores/Val - DICE Score, class {}'.format(c)] = acc[i]
                
                logger.info('Evaluation result: {}, val loss: {}'.format(eval_result, val_loss))
                
                for key, acc in eval_result.items():
                    summary_writer.add_scalar(key, acc, global_step=global_step)
                summary_writer.add_scalar('losses/Validation loss', val_loss, global_step=global_step)
                if lowest_loss - val_loss > cfg.TEST.EARLY_STOPPING_TOL:
                    lowest_loss = val_loss
                    early_stopping_count = 0
                    is_best_cp = True
                else:
                    early_stopping_count += 1
                    is_best_cp = False

                if early_stopping_count >= cfg.TEST.EARLY_STOPPING_COUNT: #øker den til 50
                    logger.info('Early stopping at epoch {}'.format(epoch))
                    is_early_stopping = True

            model.train(True)  # *IMPORTANT*: change to train mode after eval.
            checkpointer.save("model_{:03d}".format(epoch), is_best_cp=is_best_cp, **arguments)
        if cfg.FIND_LR_EPOCH > 0 and epoch % cfg.FIND_LR_EPOCH == 0 and epoch > 0:
            logger.info('Finding new LR')
            
            lr_finder.range_test(train_data_loader, end_lr=100, num_iter=100)
            lr_finder.plot()
            lr_finder.reset()
            plot_path = os.path.join(cfg.OUTPUT_DIR, 'lr_finder')
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            plt.savefig(plot_path + '/epoch{}.png'.format(epoch))
        start_iter = iteration

        
        running_hours = int(arguments['running_time']/3600)
        running_minutes = int(arguments['running_time']/60) - running_hours*60
        running_seconds = arguments['running_time'] - running_minutes*60
        logger.info('Running training for: {:02d} hours, {:02d} minutes and {:02d} seconds'.format(running_hours, running_minutes, int(running_seconds)))
        

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model
