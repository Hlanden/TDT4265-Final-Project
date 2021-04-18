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
    
    '''
    v_predb = predb.clone()
    v_yb = yb.clone()
    v_predb2 = predb.clone()
    v_yb2 = yb.clone()
    '''

    predb = predb.view(-1)
    yb = yb.view(-1)
    

    for i in range(1,num_classes+1):
        class_predb = torch.where(predb == i, 1,0) 
        class_yb = torch.where(yb == i, 1,0)

        intersection = (class_predb * class_yb).sum()
        class_dice = (2*intersection)/(class_predb.sum() + class_yb.sum() + smooth) 
        class_dice_np = class_dice.clone().cpu()               
        dice[0][i-1] = class_dice_np.numpy()
    
    
    '''
    v_predb = torch.where(v_predb == 1, 1,0) 
    v_yb = torch.where(v_yb == 1, 1,0)
    x = v_predb[0].clone().cpu() 
    y = v_yb[0].clone().cpu() 
    x = x.numpy()
    y = y.numpy()


    v_predb2 = torch.where(v_predb2 == 2, 1,0) 
    v_yb2 = torch.where(v_yb2 == 2, 1,0)
    x2 = v_predb2[0].clone().cpu() 
    y2 = v_yb2[0].clone().cpu() 
    x2 = x2.numpy()
    y2 = y2.numpy()
    #print(x)
    fig, ax = plt.subplots(1,4)
    ax[0].imshow(x.squeeze())
    ax[1].imshow(y.squeeze())
    ax[2].imshow(x2.squeeze())
    ax[3].imshow(y2.squeeze())
    plt.savefig('pic.png')
    plt.show()
    ''' 
    return dice