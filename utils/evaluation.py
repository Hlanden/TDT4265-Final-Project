import numpy as np
import torch
import matplotlib.pyplot as plt
from data.transforms import Resize
import cv2

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_score(predb, yb):
    predb = predb.argmax(dim=1)
    predb = predb.view(-1)
    yb = yb.view(-1)
    intersection = (predb * yb).sum()                      
    dice = (2*intersection)/(predb.sum() + yb.sum())
    return dice

def dice_score_multiclass(predb, yb, num_classes, shapes=None, padding=None, smooth = 0.00001,org_targets=None):   #num_classes should not include background
    if type(predb) == torch.Tensor:
        predb = predb.cpu().detach().numpy()
    if type(yb) == torch.Tensor:
        yb = yb.cpu().detach().numpy()
    
    predb = np.argmax(predb, axis=1)
    reshaped_predb = []
    reshaped_yb = []
    if shapes is not None and padding is not None:
        if org_targets is not None:
            for shape, padding, p, y, org_target in zip(shapes, padding, predb, yb, org_targets):
                x_org, y_org = y.shape
                x_lim = x_org - padding[0]
                y_lim = y_org - padding[1]

                reshaped_predb.extend(cv2.resize(p[:x_lim, :y_lim].astype(np.float32), dsize=tuple(reversed(shape)), interpolation=cv2.INTER_LINEAR).flat)
                reshaped_yb.extend(org_target.flat)
        else:
            for shape, padding, p, y in zip(shapes, padding, predb, yb):
                x_org, y_org = y.shape
                x_lim = x_org - padding[0]
                y_lim = y_org - padding[1]

                reshaped_predb.extend(cv2.resize(p[:x_lim, :y_lim].astype(np.float32), dsize=tuple(reversed(shape)), interpolation=cv2.INTER_LINEAR).flat)
                reshaped_yb.extend(cv2.resize(y[:x_lim, :y_lim].astype(np.float32), dsize=tuple(reversed(shape)), interpolation=cv2.INTER_LINEAR).flat)

        predb = np.array(reshaped_predb)
        
        yb = np.array(reshaped_yb)
            
    
    

    dice = np.zeros((1,num_classes))
    

    #predb = predb.view(-1)
    #yb = yb.view(-1)

    predb = predb.flat
    yb = yb.flat


    

    for i in range(1,num_classes+1):
        class_predb = np.where(predb == i, 1,0) 
        class_yb = np.where(yb == i, 1,0)

        intersection = (class_predb * class_yb).sum()
        class_dice = (2*intersection)/(class_predb.sum() + class_yb.sum() + smooth) 
        #class_dice_np = class_dice.clone().cpu()               
        #dice[0][i-1] = class_dice_np.numpy()
        dice[0][i-1] = class_dice
        

    return dice

