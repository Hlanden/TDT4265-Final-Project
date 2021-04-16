import numpy as np

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_score(predb, yb):
    predb = predb.argmax(dim=1)
    predb = predb.view(-1)
    yb = yb.view(-1)
    intersection = (predb * yb).sum()                      
    dice = (2*intersection)/(predb.sum() + yb.sum())
    return dice

def dice_score_multiclass(predb, yb, num_classes):   #num_classes should not include background

    dice = np.zeros((1,num_classes))
    predb = predb.argmax(dim=1)
    predb = predb.view(-1)
    yb = yb.view(-1)
    for i in range(1,num_classes):
        class_predb = np.where(predb == i, 1,0) 
        class_yb = np.where(yb == i, 1,0)
        intersection = (class_predb * class_yb).sum()
        class_dice = (2*intersection)/(predb.sum() + yb.sum() + smooth)                 
        dice[i-1] = class_dice.copy()
    return dice