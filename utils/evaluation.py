import numpy as np
import torch
import matplotlib.pyplot as plt

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_score(predb, yb):
    predb = predb.argmax(dim=1)
    predb = predb.view(-1)
    yb = yb.view(-1)
    intersection = (predb * yb).sum()                      
    dice = (2*intersection)/(predb.sum() + yb.sum())
    return dice

def dice_score_multiclass(predb, yb, num_classes, model, smooth = 0.00001):   #num_classes should not include background

    dice = np.zeros((1,num_classes))
    predb = predb.argmax(dim=1)

    predb = predb.view(-1)
    yb = yb.view(-1)
    

    for i in range(1,num_classes+1):
        class_predb = torch.where(predb == i, 1,0) 
        class_yb = torch.where(yb == i, 1,0)

        intersection = (class_predb * class_yb).sum()
        class_dice = (2*intersection)/(class_predb.sum() + class_yb.sum() + smooth) 
        class_dice_np = class_dice.clone().cpu()               
        dice[0][i-1] = class_dice_np.numpy()
    

    return dice

