import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)

def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel)*1.0
    return skel

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Expects probabilities [B, C, H, W]
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        
        # Calculate precision and sensitivity
        tprec = (torch.sum(torch.mul(skel_pred, y_true)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.mul(skel_true, y_pred)) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice

class BCE_clDice_Loss(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1.):
        """
        Combination of BCE and soft clDice loss.
        alpha: weight for BCE loss. (1 - alpha) is for clDice.
        """
        super(BCE_clDice_Loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cldice = soft_cldice(iter_=iter_, smooth=smooth)
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] network output before sigmoid
        targets: [B, C, H, W] binary masks
        """
        bce_loss = self.bce(logits, targets)
        
        # apply sigmoid to get probabilities for cldice
        probs = torch.sigmoid(logits)
        cldice_loss = self.cldice(probs, targets)
        
        return self.alpha * bce_loss + (1. - self.alpha) * cldice_loss
