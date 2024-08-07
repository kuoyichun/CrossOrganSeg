import torch
from torch import nn

class Charbonnier_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
    
from torchmetrics.image import StructuralSimilarityIndexMeasure
class SSIM_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, X, Y):
        return 1 - self.SSIM(X, Y)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).type_as(x)

    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.ones(x.size()).type_as(x)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask

def warp_seg(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).type_as(x)

    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    # output = torch.where(output > 0.5, 1, 0) #這一行要弄掉
    mask = torch.ones(x.size()).type_as(x)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask

import flow_vis
import numpy as np
def visual_flow(flows):
    flows = flows.cpu().detach().numpy()
    flows = [flow_vis.flow_to_color(flow.transpose(1, 2, 0)) for flow in flows]
    return np.array(flows, dtype='uint8')

# Metric
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
class MeasureMetric(pl.LightningModule):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics
        for m in metrics:
            setattr(self, m, MeanMetric().to('cuda'))

        if 'PSNR' in metrics:
            self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        if 'SSIM' in metrics:
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def update(self, pred, target):
        if 'PSNR' in self.metrics:
            psnr = self.psnr(pred, target)
            self.PSNR.update(psnr)
           
        if 'SSIM' in self.metrics:
            ssim = self.ssim(pred, target)
            self.SSIM.update(ssim)
            
        if 'DICE' in self.metrics:
            d = dice(pred, target)
            self.DICE.update(d)
            

    def compute(self):
        results = {}
        for m in self.metrics:
            meanmetric = getattr(self, m)
            results[m] = meanmetric.compute()
            

        return results
    
    def reset(self):
        for m in self.metrics:
            meanmetric = getattr(self, m)
            meanmetric.reset()
    
    

def dice(pred, target):
    smooth = 1e-5
    B = pred.shape[0]
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Dice loss
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return 1 - dice(pred, target)
