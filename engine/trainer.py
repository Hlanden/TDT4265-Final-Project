import collections
import datetime
import logging
import os
import time
import torch
from torch.utils.data import dataloader
import torch.utils.tensorboard
from engine.inference import do_evaluation
from utils.metric_logger import MetricLogger
from utils import torch_utils



def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)

def do_train(cfg, model,
             data_loader,
             optimizer,
             checkpointer,
             arguments,
             loss_fn):
    logger = logging.getLogger("UNET.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()

    summary_writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    

    while (time.time() - start_training_time)/60 <= cfg.SOLVER.MAX_MINUTES:        
        for iteration, (images, targets) in enumerate(data_loader, start_iter):
            iteration = iteration + 1
            arguments["iteration"] = iteration
            
            
            images = torch_utils.to_cuda(images)
            targets = torch_utils.to_cuda(targets)
            x = model(images)
            
            loss = loss_fn(x, targets.long())

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
                    'losses/total_loss', loss, global_step=global_step)
                
                summary_writer.add_scalar(
                    'lr', optimizer.param_groups[0]['lr'],
                    global_step=global_step)

            if iteration % cfg.MODEL_SAVE_STEP == 0:
                checkpointer.save("model_{:06d}".format(iteration), **arguments)
                

            # TODO: Currently deactivated. Need dataloader class to make eval
            """if cfg.EVAL_STEP > 0 and iteration % cfg.EVAL_STEP == 0:
                #eval_results = do_evaluation(cfg, model, iteration=iteration)
                eval_results = [0] 
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metric(
                        eval_result['metrics'], 'metrics/' + dataset,summary_writer, iteration)
                model.train()  # *IMPORTANT*: change to train mode after eval."""

            if iteration >= cfg.SOLVER.MAX_ITER:
                break
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
