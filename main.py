from typing import Any
from data.dataset import FlowDataset, FusionDataset, Fusion_test_dataset
from data.dataset_1organ import FusionDataset1Organ
from models import Flow, Refine, DynFilter, DynFilter_poscat, weight_combined, correlation
import tools


import argparse
import yaml
import time
import os, sys
from tqdm import trange
import numpy as np
import shutil

from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
import torch



import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

def parser():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # Mode
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--fast_run', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_auto', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # YAML path
    parser.add_argument('--yaml_path', required=True)
    # checkpoint path
    parser.add_argument('--ckpt_path', required='--test' in sys.argv)
    # task
    parser.add_argument('--task', help='flow or fusion or fusion_org', required=True)
    # test or test_auto都需要輸入patient
    parser.add_argument('--patient', help='patient name', required='--test' in sys.argv or '--test_auto' in sys.argv)
    # test_auto
    parser.add_argument('--label', type=int, help='label', required='--test_auto' in sys.argv)
    parser.add_argument('--list_path', help='list path', required='--test_auto' in sys.argv)
    

    return parser.parse_args()

class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        # self.automatic_optimization = False
        self.cfg = cfg

        self.Flow = Flow.Model(cfg)

        self.char_loss = tools.Charbonnier_loss()
        # self.ssim_loss = tools.SSIM_loss()
        self.gdl_loss = tools.GDL_loss()

        self.metrics = metrics = ['PSNR', 'SSIM']
        self.MM0 = tools.MeasureMetric(metrics=metrics)
        self.MM1 = tools.MeasureMetric(metrics=metrics)
        self.MM0val = tools.MeasureMetric(metrics=metrics)
        self.MM1val = tools.MeasureMetric(metrics=metrics)
        self.MM01val = tools.MeasureMetric(metrics=metrics)

    def training_step(self, batch, batch_idx):
        ct0, ct1 = batch

        flows10 = self.Flow(ct0, ct1)

        char_losses = []
        # gdl_losses = []
        # ssim_losses = []
        I1_0s = []
        for i in range(1, self.cfg['conv_layers']+1):
            if i == 1:
                I0, I1 = ct0, ct1
            else:
                I0 = F.interpolate(I0, scale_factor=0.5, mode='bilinear', align_corners=True)
                I1 = F.interpolate(I1, scale_factor=0.5, mode='bilinear', align_corners=True)

            # warp
            I1_0 = tools.warp(I0, flows10[-i])
            # I0_1 = tools.warp(I1, flows01[-i])

            I1_0s.append(I1_0)
            # I0_1s.append(I0_1)

            # char = self.char_loss(I0_1, I0) + self.char_loss(I1_0, I1)
            char = self.char_loss(I1_0, I1)
            char_losses.append(char)

            # gdl = self.gdl_loss(I0_1, I0) + self.gdl_loss(I1_0, I1)
            # gdl_losses.append(gdl)
            # ssim = self.ssim_loss(I0_1, I0) + self.ssim_loss(I1_0, I1)
            # ssim_losses.append(ssim)

        char_loss = sum(char_losses)
        loss = char_loss
        # ssim_loss = sum(ssim_losses)
        # gdl_loss = sum(gdl_losses)
        # loss = char_loss + gdl_loss

        # Log
        self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.logger.experiment.add_scalar('Train/char_loss', char_loss, self.global_step)
        # self.logger.experiment.add_scalar('Train/gdl_loss', gdl_loss, self.global_step)
        show_num = self.cfg['show_num']
        if batch_idx % self.cfg['show_step'] == 0:
            self.logger.experiment.add_images('Train/ct0', ct0[:show_num], self.global_step)
            self.logger.experiment.add_images('Train/ct1', ct1[:show_num], self.global_step)
            # self.logger.experiment.add_images('Train/ct0_1', I0_1s[0][:show_num], self.global_step)
            self.logger.experiment.add_images('Train/ct1_0', I1_0s[0][:show_num], self.global_step)
            # self.logger.experiment.add_images('Train/flow01', tools.visual_flow(flows01[-1])[:show_num], self.global_step, dataformats='NHWC')
            self.logger.experiment.add_images('Train/flow10', tools.visual_flow(flows10[-1])[:show_num], self.global_step, dataformats='NHWC')

        return loss
    
    # # 多張間距ct
    # def training_step(self, batch, batch_idx):
    #     ctA, ctBs = batch
    #     for ctB in ctBs:
            
    #         flowsBA = self.Flow(ctA, ctB)

    #         char_losses = []
    #         IB_As = []
    #         for i in range(1, self.cfg['conv_layers']+1):
    #             if i == 1:
    #                 IA, IB = ctA, ctB
    #             else:
    #                 IA = F.interpolate(IA, scale_factor=0.5, mode='bilinear', align_corners=True)
    #                 IB = F.interpolate(IB, scale_factor=0.5, mode='bilinear', align_corners=True)

    #             # warp
    #             IB_A = tools.warp(IA, flowsBA[-i])
    #             IB_As.append(IB_A)

    #             char = self.char_loss(IB_A, IB)
    #             char_losses.append(char)

    #         char_loss = sum(char_losses)
    #         loss = char_loss

    #         # Log
    #         self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    #         self.logger.experiment.add_scalar('Train/char_loss', char_loss, self.global_step)
    #         show_num = self.cfg['show_num']
    #         if batch_idx % self.cfg['show_step'] == 0:
    #             self.logger.experiment.add_images('Train/ctA', ctA[:show_num], self.global_step)
    #             self.logger.experiment.add_images('Train/ctB', ctB[:show_num], self.global_step)
    #             self.logger.experiment.add_images('Train/ctB_A', IB_As[0][:show_num], self.global_step)
    #             self.logger.experiment.add_images('Train/flowBA', tools.visual_flow(flowsBA[-1])[:show_num], self.global_step, dataformats='NHWC')

    #     return loss
    
    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        ct0, ct1 = batch

        # flows01, flows10 = self.Flow(ct0, ct1)
        flows10 = self.Flow(ct0, ct1)

        # only origin size
        I0, I1 = ct0, ct1
        # warp
        I1_0 = tools.warp(I0, flows10[-1])
        # I0_1 = tools.warp(I1, flows01[-1])

        show_num = self.cfg['show_num']
        self.tmp = {
            'ct0': ct0[:show_num],
            'ct1': ct1[:show_num],
            # 'ct0_1': I0_1[:show_num],
            'ct1_0': I1_0[:show_num],
        }

        # metric
        # self.MM0val.update(I0_1, I0)
        self.MM1val.update(I1_0, I1)
        self.MM01val.update(I0, I1)

    def on_validation_epoch_end(self):
        self.logger.experiment.add_images('Val/ct0', self.tmp['ct0'], self.current_epoch)
        self.logger.experiment.add_images('Val/ct1', self.tmp['ct1'], self.current_epoch)
        # self.logger.experiment.add_images('Val/ct0_1', self.tmp['ct0_1'], self.current_epoch)
        self.logger.experiment.add_images('Val/ct1_0', self.tmp['ct1_0'], self.current_epoch)
        # metric
        # MM0val = self.MM0val.compute()
        MM1val = self.MM1val.compute()
        MM01val = self.MM01val.compute()
        # self.MM0val.reset()
        self.MM1val.reset()
        self.MM01val.reset()
        for m in self.metrics:
            # self.logger.experiment.add_scalar(f'Val/{m}/I0', MM0val[m], self.current_epoch)
            self.logger.experiment.add_scalar(f'Val/{m}/I1', MM1val[m], self.current_epoch)
            self.logger.experiment.add_scalar(f'Val/{m}/I01', MM01val[m], self.current_epoch)
            print('gen:', m, MM1val[m])
            print('GT difference:', m, MM01val[m])

    def test_step(self, batch, batch_idx):
        ct0, ct1 = batch

        flows01, flows10 = self.Flow(ct0, ct1)

        # only origin size
        I0, I1 = ct0, ct1
        # warp
        I1_0 = tools.warp(I0, flows10[-1])
        I0_1 = tools.warp(I1, flows01[-1])

        # metric
        self.MM0.update(I0_1, I0)
        self.MM1.update(I1_0, I1)

    def on_test_end(self):
        MM0 = self.MM0.compute()
        MM1 = self.MM1.compute()

        for m in self.metrics:
            print(m)
            print("I0: {:.2f}".format(MM0[m]))
            print("I1: {:.2f}".format(MM1[m]))

    def forward(self, ct0, ct1):
        flows10 = self.Flow(ct0, ct1)
        return flows10

    def configure_optimizers(self):
        opt = optim.Adam(self.Flow.parameters(), lr=self.cfg['lr'])
        sch = optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.9)
        return [opt], [sch]

class FusionModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        # self.automatic_optimization = False
        self.cfg = cfg

        self.Flow = LitModel.load_from_checkpoint(cfg['flow_ckpt_path']).eval()
        self.Refine = Refine.Model(cfg)

        self.char_loss = tools.Charbonnier_loss()

    def iteratively_warp(self, seg, batch_ct0, batch_ct1):
        batch_ct0 = torch.stack(batch_ct0, dim=0).squeeze(1)
        batch_ct1 = torch.stack(batch_ct1, dim=0).squeeze(1)

        with torch.no_grad():
            flows10 = self.Flow(batch_ct0, batch_ct1)
        flow10 = flows10[-1]
        del flows10

        # for flow in batch, iteratly warp
        for f10 in flow10:
            seg = tools.warp_seg(seg, f10)
            

        return seg
    
    def configure_batch(self, seg, cts):
        batch_ct0 = []
        batch_ct1 = []
        for i in range(len(cts)-1):
            ct0 = cts[i]
            ct1 = cts[i+1]

            batch_ct0.append(ct0)
            batch_ct1.append(ct1)

            if len(batch_ct0) == self.cfg['batch_size']:
                seg = self.iteratively_warp(seg, batch_ct0, batch_ct1)
                batch_ct0 = []
                batch_ct1 = []
        if len(batch_ct0) != 0:
            seg = self.iteratively_warp(seg, batch_ct0, batch_ct1)
        return seg
        

    def training_step(self, batch, batch_idx):
        forward_cts, backward_cts, seg0, seg_target, seg25, ratio = batch
        
        # forward, configure batch
        seg_forward = seg0
        seg_forward = self.configure_batch(seg_forward, forward_cts)
        
        # backward, configure batch
        seg_backward = seg25
        backward_cts = list(reversed(backward_cts))
        seg_backward = self.configure_batch(seg_backward, backward_cts)

        seg_union = seg_forward * (1 - ratio) + seg_backward * ratio
        seg_before_refine = torch.where(seg_union >= 0.5, 1, 0)

        ct_target = forward_cts[-1]
        target = ct_target * seg_target
        ct_union = seg_union * ct_target

        seg_residual = self.Refine(ct_union) 
        seg_union = seg_union + seg_residual
        ct_union = seg_union * ct_target

        # loss
        char_loss = self.char_loss(ct_union, target)
        loss = char_loss

        # Log
        seg_union_show = torch.where(seg_union >= 0.5, 1, 0)
        self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.logger.experiment.add_scalar('Train/char_loss', char_loss, self.global_step)
        if batch_idx % self.cfg['show_step'] == 0:
            self.logger.experiment.add_image('Train/seg0', seg0[0], self.global_step)
            self.logger.experiment.add_image('Train/seg25', seg25[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_target', seg_target[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_forward', seg_forward[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_backward', seg_backward[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_union', seg_union_show[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_residual', seg_residual[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_before_refine', seg_before_refine[0], self.global_step)

        return loss


    def test_step(self, batch, batch_idx):
        ct0, ct1, self.segs, index_paths = batch
        self.marked = [_[0] for _ in index_paths]

        
        with torch.no_grad():
            # forward
            forward_flow = self.Flow(ct0, ct1)[-1]
            # backward
            backward_flow = self.Flow(ct1, ct0)[-1]
        
        if batch_idx == 0:
            self.forward_flows = forward_flow
            self.backward_flows = backward_flow
        else:
            self.forward_flows = torch.cat([self.forward_flows, forward_flow], dim=0)
            self.backward_flows = torch.cat([self.backward_flows, backward_flow], dim=0)

        del forward_flow
        del backward_flow
    
    def on_test_end(self):
        self.segs = [_[0] for _ in self.segs]

        segs = {}
        j = 0
        for i in range(len(self.forward_flows)+1):
            if i in self.marked:
                segs[i] = self.segs[j]
                j += 1
            else:
                segs[i] = None

        for idx in range(len(self.forward_flows)):
            if idx in self.marked:
                f_seg = segs[idx].unsqueeze(0)
            else:
                # forward
                f_seg = tools.warp_seg(f_seg, self.forward_flows[idx-1])
                segs[idx] = [f_seg]

        for idx in range(len(self.backward_flows), 0, -1):
            if idx in self.marked:
                b_seg = segs[idx].unsqueeze(0)
            else:
                # backward
                b_seg = tools.warp_seg(b_seg, self.backward_flows[idx])
                segs[idx].append(b_seg)
            
        import numpy as np
        import shutil
        shutil.rmtree(self.cfg['save_folder'], ignore_errors=True)
        
        forward_save_folder = os.path.join(self.cfg['save_folder'], 'forward')
        backward_save_folder = os.path.join(self.cfg['save_folder'], 'backward')
        before_save_folder = os.path.join(self.cfg['save_folder'], 'before')
        union_save_folder = os.path.join(self.cfg['save_folder'], 'union')
        os.makedirs(forward_save_folder)
        os.makedirs(backward_save_folder)
        os.makedirs(before_save_folder)
        os.makedirs(union_save_folder)
        for k, v in segs.items():
            if type(v) is list: # first of list is forward, second is backward
                fv = v[0][0, 0].cpu().numpy()
                np.save(os.path.join(forward_save_folder, 'seg{:03d}.npy'.format(k)), fv)
                bv = v[1][0, 0].cpu().numpy()
                np.save(os.path.join(backward_save_folder, 'seg{:03d}.npy'.format(k)), bv)
            else:
                v = v[0].cpu().numpy()
                np.save(os.path.join(forward_save_folder, 'seg{:03d}*.npy'.format(k)), v)
                np.save(os.path.join(backward_save_folder, 'seg{:03d}*.npy'.format(k)), v)

        keys = list(segs.keys())
        keys.sort()
        j = 0
        for k in keys[:-1]:
            if k in self.marked:
                start = self.marked[j]
                end = self.marked[j+1]
                length = end - start + 1
                np.save(os.path.join(before_save_folder, 'before_seg_union{:03d}*.npy'.format(start)), self.segs[j][0].cpu().numpy())
                np.save(os.path.join(union_save_folder, 'seg_union{:03d}*.npy'.format(start)), self.segs[j][0].cpu().numpy())
                np.save(os.path.join(before_save_folder, 'before_seg_union{:03d}*.npy'.format(end)), self.segs[j+1][0].cpu().numpy())
                np.save(os.path.join(union_save_folder, 'seg_union{:03d}*.npy'.format(end)), self.segs[j+1][0].cpu().numpy())
                j += 1
            else:
                ratio = (k - start) / (length - 1)
                f, b = segs[k]
                seg_union = f * (1 - ratio) + b * ratio
                before_seg_union = torch.where(seg_union >= 0.5, 1, 0)
                np.save(os.path.join(before_save_folder, 'before_seg_union{:03d}.npy'.format(k)), before_seg_union[0, 0].cpu().numpy())
                seg_residual = self.Refine(seg_union)
                seg_union = seg_union + seg_residual
                seg_union = torch.where(seg_union >= 0.5, 1, 0)
                np.save(os.path.join(union_save_folder, 'seg_union{:03d}.npy'.format(k)), seg_union[0, 0].cpu().numpy())

    def configure_optimizers(self):
        opt = optim.Adam(self.Refine.parameters(), lr=self.cfg['lr'])
        sch = optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.9)
        return[opt], [sch]

class DynFilterModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        # self.automatic_optimization = False
        self.cfg = cfg
        self.Flow = LitModel.load_from_checkpoint(cfg['flow_ckpt_path'])
        self.Flow.eval()
        self.Flow.freeze()
        
        self.DynFilter_model = DynFilter.DFNet(cfg)

        # self.DynFilter_model = DynFilter_poscat.DFNet(cfg)
        self.filtering = DynFilter.dynFilter()
        # self.filtering = DynFilter_poscat.dynFilter()
        self.PE = tools.PositionEncoding(shape=[cfg['resolution'], cfg['resolution']])
        self.L1loss = nn.L1Loss()
        self.BCEloss = nn.BCELoss()

    def iteratively_warp(self, seg, batch_ct0, batch_ct1):
        batch_ct0 = torch.stack(batch_ct0, dim=0).squeeze(1)
        batch_ct1 = torch.stack(batch_ct1, dim=0).squeeze(1)

        with torch.no_grad():
            flows10 = self.Flow(batch_ct0, batch_ct1)
        flow10 = flows10[-1]
        del flows10

        # for flow in batch, iteratly warp
        
        for f10 in flow10:
            seg = tools.warp_seg(seg, f10)
        
        return seg
    
    def configure_batch(self, seg, cts):
        batch_ct0 = []
        batch_ct1 = []
        for i in range(len(cts)-1):
            ct0 = cts[i]
            ct1 = cts[i+1]

            batch_ct0.append(ct0)
            batch_ct1.append(ct1)

            if len(batch_ct0) == self.cfg['batch_size']:
                seg = self.iteratively_warp(seg, batch_ct0, batch_ct1)
                batch_ct0 = []
                batch_ct1 = []
        if len(batch_ct0) != 0:
            seg = self.iteratively_warp(seg, batch_ct0, batch_ct1)
        return seg
    
    def training_step(self, batch, batch_idx):
        self.Flow.freeze()
        forward_cts, backward_cts, seg0, seg_target, seg25, ratio, global_ratio, ct_t = batch
        
        # forward, configure batch
        seg_forward = seg0
        seg_forward = self.configure_batch(seg_forward, forward_cts)
        
        # backward, configure batch
        seg_backward = seg25
        backward_cts = list(reversed(backward_cts))
        seg_backward = self.configure_batch(seg_backward, backward_cts)
        seg_union = seg_forward * (1 - ratio) + seg_backward * ratio
        seg_weight_combined = torch.where(seg_union >= 0.5, 1, 0)
        
        pos_ecd = self.PE(ratio, mode='train').type_as(seg_forward) #position encoding
        
        
        DF_input = torch.cat([seg_forward, seg_backward, pos_ecd], dim=1)
        candidates = torch.cat([seg_forward, seg_backward], dim=1)
        
        filter = self.DynFilter_model(DF_input)
        ##### For 240313_onlyDFnet #####
        self.fused = self.filtering(candidates, filter)

        # model改成output seg_residual
        ##### For 240104 #####
        # seg_residual = self.filtering(candidates, filter)
        # self.fused = seg_union + seg_residual
        
        
        self.fused_thred = torch.where(self.fused >= 0.5, 1, 0)
        # ct_fused_region = ct_t * self.fused_thred #預測區域用濾過的fused算
        # ct_gt_region = ct_t * seg_target

        # loss
        L1_loss = self.L1loss(self.fused, seg_target)
        # ct_L1_loss = self.L1loss(ct_fused_region, ct_gt_region)
        print('L1_loss:', L1_loss)
        # print('ct_L1_loss:', ct_L1_loss)
        # loss = L1_loss + ct_L1_loss #加入預測的ct區域一起算loss
        loss = L1_loss

        # Log
        seg_fused = torch.where(self.fused >= 0.5, 1, 0)
        self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.logger.experiment.add_scalar('Train/L1_loss', L1_loss, self.global_step)
        # self.logger.experiment.add_scalar('Train/ct_L1_loss', ct_L1_loss, self.global_step)
        # self.logger.experiment.add_scalar('Train/Total_loss', loss, self.global_step)
        if batch_idx % self.cfg['show_step'] == 0:
            self.logger.experiment.add_image('Train/seg0', seg0[0], self.global_step)
            self.logger.experiment.add_image('Train/seg25', seg25[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_target', seg_target[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_forward', seg_forward[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_backward', seg_backward[0], self.global_step)
            # self.logger.experiment.add_image('Train/seg_weight_combined', seg_weight_combined[0], self.global_step)
            # self.logger.experiment.add_image('Train/seg_residual', seg_residual[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_fused', seg_fused[0], self.global_step)
            # self.logger.experiment.add_image('Train/ct_fused_region', ct_fused_region[0], self.global_step)
            # self.logger.experiment.add_image('Train/ct_gt_region', ct_gt_region[0], self.global_step)

        return loss

    
       
    def test_step(self, batch, batch_idx):
        ct0, ct1, self.segs, index_paths, self.cts, self.all_cts, self.all_segs = batch #self.cts, self.all_cts 為了存CT
        self.marked = [_[0] for _ in index_paths]
        self.Flow.eval()
        with torch.no_grad():
            # forward
            forward_flow = self.Flow(ct0, ct1)[-1]
            # backward
            backward_flow = self.Flow(ct1, ct0)[-1]

        if batch_idx == 0:
            self.forward_flows = forward_flow
            self.backward_flows = backward_flow
        else:
            self.forward_flows = torch.cat([self.forward_flows, forward_flow], dim=0)
            self.backward_flows = torch.cat([self.backward_flows, backward_flow], dim=0)

        del forward_flow
        del backward_flow

    def on_test_end(self):
        self.segs = [_[0] for _ in self.segs]
        self.cts  = [_[0] for _ in self.cts] #為了存CT
        

        # initialize saving dictionary
        # ct, None, None, None, ..., ct, None, None, None, ..., ct
        segs = {}
        cts = {} #為了存CT
        gen_f_cts = {} #為了存CT
        gen_b_cts = {} #為了存CT
        gen_f_segs = {}
        gen_b_segs = {}
        j = 0
        for i in range(len(self.forward_flows)+1):
            if i in self.marked:
                segs[i] = self.segs[j]
                cts[i] = self.cts[j] #為了存CT
                gen_f_cts[i] = self.cts[j] #為了存CT
                gen_b_cts[i] = self.cts[j] #為了存CT
                gen_f_segs[i] = self.segs[j]
                gen_b_segs[i] = self.segs[j]
                j += 1
            else:
                segs[i] = None
                cts[i] = None #為了存CT
        start_time = time.time()
        # forward
        for idx in range(len(self.forward_flows)):
            if idx in self.marked:
                f_ct = cts[idx].unsqueeze(0).unsqueeze(0) #為了存CT
                f_seg = segs[idx].unsqueeze(0)
            else:
                # forward
                f_ct = tools.warp(f_ct, self.forward_flows[idx-1]) #為了存CT
                f_seg = tools.warp_seg(f_seg, self.forward_flows[idx-1])
                segs[idx] = [f_seg]
                cts[idx] = [f_ct] #為了存CT

                gen_f_ct = tools.warp(self.all_cts[idx-1].unsqueeze(0), self.forward_flows[idx-1]) #為了存CT
                gen_f_cts[idx] = gen_f_ct[0, 0] #為了存CT
                gen_f_seg = tools.warp_seg(self.all_segs[idx-1], self.forward_flows[idx-1]) #存每步用gt傳遞的seg
                gen_f_segs[idx] = gen_f_seg[0, 0]

        # backward
        for idx in range(len(self.backward_flows), 0, -1):
            if idx in self.marked:
                b_ct = cts[idx].unsqueeze(0).unsqueeze(0) #為了存CT
                b_seg = segs[idx].unsqueeze(0)
            else:
                # backward
                b_ct = tools.warp(b_ct, self.backward_flows[idx]) #為了存CT
                b_seg = tools.warp_seg(b_seg, self.backward_flows[idx])
                segs[idx].append(b_seg)
                cts[idx].append(b_ct) #為了存CT

                gen_b_ct = tools.warp(self.all_cts[idx+1].unsqueeze(0), self.backward_flows[idx]) #為了存CT
                gen_b_cts[idx] = gen_b_ct[0, 0] #為了存CT
                gen_b_seg = tools.warp_seg(self.all_segs[idx+1], self.backward_flows[idx])
                gen_b_segs[idx] = gen_b_seg[0, 0]

        shutil.rmtree(self.cfg['save_folder'], ignore_errors=True)
        
        forward_save_folder = os.path.join(self.cfg['save_folder'], 'forward')
        backward_save_folder = os.path.join(self.cfg['save_folder'], 'backward')
        union_save_folder = os.path.join(self.cfg['save_folder'], 'union')
        union_show_save_folder = os.path.join(self.cfg['save_folder'], 'union_show')
        residual_save_folder = os.path.join(self.cfg['save_folder'], 'residual')
        fused_save_folder = os.path.join(self.cfg['save_folder'], 'fused')

        ct_forward_save_folder = os.path.join(self.cfg['save_folder'], 'ct_forward') #為了存CT
        ct_backward_save_folder = os.path.join(self.cfg['save_folder'], 'ct_backward') #為了存CT
        gen_f_ct_save_folder = os.path.join(self.cfg['save_folder'], 'gen_f_ct') #為了存CT
        gen_b_ct_save_folder = os.path.join(self.cfg['save_folder'], 'gen_b_ct') #為了存CT
        gen_f_segs_save_folder = os.path.join(self.cfg['save_folder'], 'gen_f_segs')
        gen_b_segs_save_folder = os.path.join(self.cfg['save_folder'], 'gen_b_segs')

        os.makedirs(gen_f_ct_save_folder) #為了存CT
        os.makedirs(gen_b_ct_save_folder) #為了存CT
        os.makedirs(forward_save_folder)
        os.makedirs(backward_save_folder)
        os.makedirs(union_save_folder)
        os.makedirs(union_show_save_folder)
        os.makedirs(residual_save_folder)
        os.makedirs(fused_save_folder)
        os.makedirs(ct_forward_save_folder) #為了存CT
        os.makedirs(ct_backward_save_folder) #為了存CT
        os.makedirs(gen_f_segs_save_folder)
        os.makedirs(gen_b_segs_save_folder)

        for k, v in segs.items():
            if type(v) is list: # first of list is forward, second is backward
                fv = v[0][0, 0].cpu().numpy()
                bv = v[1][0, 0].cpu().numpy()
                np.save(os.path.join(forward_save_folder, 'seg{:03d}.npy'.format(k)), fv)
                np.save(os.path.join(backward_save_folder, 'seg{:03d}.npy'.format(k)), bv)
            else:
                v = v[0].cpu().numpy()
                np.save(os.path.join(forward_save_folder, 'seg{:03d}*.npy'.format(k)), v)
                np.save(os.path.join(backward_save_folder, 'seg{:03d}*.npy'.format(k)), v)
        
        for k, v in cts.items(): #為了存CT
            if type(v) is list: # first of list is forward, second is backward
                
                fv = v[0][0, 0].cpu().numpy() #(512, 512)

                np.save(os.path.join(ct_forward_save_folder, 'ct{:03d}.npy'.format(k)), fv)
               
                bv = v[1][0, 0].cpu().numpy()
                np.save(os.path.join(ct_backward_save_folder, 'ct{:03d}.npy'.format(k)), bv)
            else:
                v = v.cpu().numpy()
                np.save(os.path.join(ct_forward_save_folder, 'ct{:03d}*.npy'.format(k)), v)
                np.save(os.path.join(ct_backward_save_folder, 'ct{:03d}*.npy'.format(k)), v)
        
        for i in range(len(gen_f_cts)):
            np.save(os.path.join(gen_f_ct_save_folder, 'ct{:03d}*.npy'.format(i)), gen_f_cts[i].cpu().numpy()) #為了存CT
            np.save(os.path.join(gen_b_ct_save_folder, 'ct{:03d}*.npy'.format(i)), gen_b_cts[i].cpu().numpy()) #為了存CT
            np.save(os.path.join(gen_f_segs_save_folder, 'seg{:03d}*.npy'.format(i)), gen_f_segs[i].cpu().numpy())
            np.save(os.path.join(gen_b_segs_save_folder, 'seg{:03d}*.npy'.format(i)), gen_b_segs[i].cpu().numpy())

        keys = list(segs.keys())
        keys.sort()
        j = 0
        for k in keys[:-1]:
            if k in self.marked: #GT
                start = self.marked[j]
                end = self.marked[j+1]
                length = end - start + 1
                np.save(os.path.join(union_save_folder, 'seg_union{:03d}*.npy'.format(start)), self.segs[j][0].cpu().numpy())
                np.save(os.path.join(union_show_save_folder, 'seg_union_show{:03d}*.npy'.format(start)), self.segs[j][0].cpu().numpy())
                np.save(os.path.join(fused_save_folder, 'seg_fused{:03d}*.npy'.format(start)), self.segs[j][0].cpu().numpy())
                
                np.save(os.path.join(union_save_folder, 'seg_union{:03d}*.npy'.format(end)), self.segs[j+1][0].cpu().numpy())
                np.save(os.path.join(union_show_save_folder, 'seg_union_show{:03d}*.npy'.format(end)), self.segs[j+1][0].cpu().numpy())
                np.save(os.path.join(fused_save_folder, 'seg_fused{:03d}*.npy'.format(end)), self.segs[j+1][0].cpu().numpy())
                j += 1
            else:
                ratio = (k - start) / (length - 1)
                f, b = segs[k]
                seg_union = f * (1 - ratio) + b * ratio
                seg_union_show = torch.where(seg_union >= 0.5, 1, 0)
                np.save(os.path.join(union_save_folder, 'seg_union{:03d}.npy'.format(k)), seg_union[0, 0].cpu().numpy())
                np.save(os.path.join(union_show_save_folder, 'seg_union_show{:03d}.npy'.format(k)), seg_union_show[0, 0].cpu().numpy())
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                pos_ecd = self.PE(ratio, mode='test').unsqueeze(0).to(device) #position encoding
                
                DF_input = torch.cat([f, b, pos_ecd], dim=1)
                candidates = torch.cat([f, b], dim=1)
                filter = self.DynFilter_model(DF_input)
                
                # model直接output self.fused 240314
                # self.fused = self.filtering(candidates, filter)
                
                # model改成output seg_residual
                
                seg_residual = self.filtering(candidates, filter)
                self.fused = seg_union + seg_residual
                
                
                #算分要用過threshold的mask
                self.fused = torch.where(self.fused >= 0.5, 1, 0)
                
                np.save(os.path.join(residual_save_folder, 'seg_residual{:03d}.npy'.format(k)), seg_residual[0, 0].cpu().numpy())
                np.save(os.path.join(fused_save_folder, 'seg_fused{:03d}.npy'.format(k)), self.fused[0, 0].cpu().numpy())
        # 算時間
        # inference_time = time.time() - start_time
        # print('inference_time:', inference_time)
        # with open('Auto_InferTime_list.txt', 'a') as f:
        #     f.write(str(round(inference_time, 4)) + '\n')
            

    def configure_optimizers(self):
        opt = optim.Adam(self.DynFilter_model.parameters(), lr=self.cfg['lr'])
        sch = optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.9)
        return[opt], [sch]

class CombinedModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.Flow = LitModel.load_from_checkpoint(cfg['flow_ckpt_path'])
        self.Flow.eval()
        self.Flow.freeze()
     
        self.combinedModel = weight_combined.CombinedModel()
        # self.correlationModel = correlation.CorrelationModel(cfg)
        self.DynFilter_model = weight_combined.DFNet(cfg)
        self.filtering = weight_combined.dynFilter()
        self.PE = tools.PositionEncoding(shape=[cfg['resolution'], cfg['resolution']])
        # self.L1loss = nn.L1Loss()
        self.Diceloss = tools.DiceLoss()

    def iteratively_warp(self, seg, batch_ct0, batch_ct1):
        batch_ct0 = torch.stack(batch_ct0, dim=0).squeeze(1)
        batch_ct1 = torch.stack(batch_ct1, dim=0).squeeze(1)

        with torch.no_grad():
            flows10 = self.Flow(batch_ct0, batch_ct1)
        flow10 = flows10[-1]
        del flows10

        # for flow in batch, iteratly warp
        
        for f10 in flow10:
            seg = tools.warp_seg(seg, f10)
        
        return seg
        
    def configure_batch(self, seg, cts):
        batch_ct0 = []
        batch_ct1 = []
        for i in range(len(cts)-1):
            ct0 = cts[i]
            ct1 = cts[i+1]

            batch_ct0.append(ct0)
            batch_ct1.append(ct1)

            if len(batch_ct0) == self.cfg['batch_size']:
                seg = self.iteratively_warp(seg, batch_ct0, batch_ct1)
                batch_ct0 = []
                batch_ct1 = []
        if len(batch_ct0) != 0:
            seg = self.iteratively_warp(seg, batch_ct0, batch_ct1)
        return seg
    
    def local_correlation_volume(self, A, B, radius):
        A_flat = A.view(A.size(0), A.size(1), -1)  # [batch_size, channels, height * width]
        B_unfolded = F.unfold(B, kernel_size=(2 * radius + 1), padding=radius)
        correlation = A_flat * B_unfolded
        correlation = correlation.view(A.size(0), (2 * radius + 1)**2, A.size(2), A.size(3))

        return correlation  
    # def training_step(self, batch, batch_idx):
    #     self.Flow.freeze()
    #     forward_cts, backward_cts, seg0, seg_target, seg25, ratio, ct_t = batch
    #     # forward, configure batch
    #     seg_forward = seg0
    #     seg_forward = self.configure_batch(seg_forward, forward_cts)
        
    #     # backward, configure batch
    #     seg_backward = seg25
    #     backward_cts = list(reversed(backward_cts))
    #     seg_backward = self.configure_batch(seg_backward, backward_cts)
    #     input_corr = torch.cat([seg_forward, seg_backward, ct_t], dim=1)
    #     seg_corr, forward_ct_region, backward_ct_region = self.correlationModel(input_corr)
    #     dice_loss = self.Diceloss(seg_corr, seg_target)
    #     loss = dice_loss

    #      # Log
    #     seg_corr  = torch.where(seg_corr >= 0.5, 1, 0)
    #     self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    #     self.logger.experiment.add_scalar('Train/Dice_loss_combined', dice_loss, self.global_step)
    #     if batch_idx % self.cfg['show_step'] == 0:
    #         self.logger.experiment.add_image('Train/seg0', seg0[0], self.global_step)
    #         self.logger.experiment.add_image('Train/seg25', seg25[0], self.global_step)
    #         self.logger.experiment.add_image('Train/seg_target', seg_target[0], self.global_step)
    #         self.logger.experiment.add_image('Train/seg_forward', seg_forward[0], self.global_step)
    #         self.logger.experiment.add_image('Train/seg_backward', seg_backward[0], self.global_step)
    #         self.logger.experiment.add_image('Train/seg_corr', seg_corr[0], self.global_step)
    #         self.logger.experiment.add_image('Train/forward_ct_region',forward_ct_region[0], self.global_step)
    #         self.logger.experiment.add_image('Train/backward_ct_region',backward_ct_region[0], self.global_step)
    #         self.logger.experiment.add_image('Train/ct_t',ct_t[0], self.global_step)
    #         self.logger.experiment.add_scalar('learning rate', torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr']), self.global_step)

    #     return loss

    def training_step(self, batch, batch_idx): #for 240129_combine+CTunion_diceloss_2weight
        self.Flow.freeze()
        forward_cts, backward_cts, seg0, seg_target, seg25, ratio, global_ratio, ct_t = batch
        # forward, configure batch
        seg_forward = seg0
        seg_forward = self.configure_batch(seg_forward, forward_cts)
        
        # backward, configure batch
        seg_backward = seg25
        backward_cts = list(reversed(backward_cts))
        seg_backward = self.configure_batch(seg_backward, backward_cts)
        seg_union = torch.where(seg_forward + seg_backward >= 1, 1, 0)
        # ct_union = seg_union * ct_t
        pos_ecd = self.PE(ratio, mode='train').type_as(seg_forward) #position encoding
        global_pos_ecd = self.PE(global_ratio, mode='train').type_as(seg_forward) #position encoding
        input_combined = torch.cat([seg_forward, seg_backward, pos_ecd, global_pos_ecd], dim=1)
        seg_combined, forward_weight, backward_weight = self.combinedModel(input_combined)

        # forward_ct_region = seg_backward * ct_t
        # backward_ct_region = seg_forward * ct_t
        
        # corr的部分
        """
        corr_volume = self.local_correlation_volume(ct_t, ct_t, self.cfg['radius'])
        input_DF = torch.cat([seg_combined, corr_volume], dim=1)
        
        filter = self.DynFilter_model(input_DF)
        residual = self.filtering(seg_combined, filter)
        
        seg_refined = seg_combined + residual

        # loss
        # print('seg_combined', seg_combined.max(), seg_combined.min())
        # print('seg_refined', seg_refined.max(), seg_refined.min())
        # dice_loss_combined = self.Diceloss(seg_combined, seg_target)
        # print('dice_loss_combined', dice_loss_combined)
        dice_loss_refined = self.Diceloss(seg_refined, seg_target)
        loss = dice_loss_refined
        # L1_loss_combined = self.L1loss(seg_combined, seg_target)
        # L1_loss_refined = self.L1loss(seg_refined, seg_target)

        # loss = L1_loss_combined + L1_loss_refined
        """
        loss = self.L1loss(seg_combined, seg_target)
        # Log
        seg_combined  = torch.where(seg_combined >= 0.5, 1, 0)
        seg_refined = torch.where(seg_refined >= 0.5, 1, 0)
        residual = (residual - residual.min()) / (residual.max() - residual.min())
        self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.logger.experiment.add_scalar('Train/Dice_loss_refined', dice_loss_refined, self.global_step)
        if batch_idx % self.cfg['show_step'] == 0:
            self.logger.experiment.add_image('Train/seg0', seg0[0], self.global_step)
            self.logger.experiment.add_image('Train/seg25', seg25[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_target', seg_target[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_forward', seg_forward[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_backward', seg_backward[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_combined', seg_combined[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_refined', seg_refined[0], self.global_step)
            # self.logger.experiment.add_image('Train/ct_union', ct_union[0], self.global_step)
            self.logger.experiment.add_image('Train/ct_t', ct_t[0], self.global_step)
            self.logger.experiment.add_image('Train/residual', residual[0], self.global_step)
            self.logger.experiment.add_image('Train/forward_weight', forward_weight[0], self.global_step)
            self.logger.experiment.add_image('Train/backward_weight', backward_weight[0], self.global_step)

        return loss
    
    def test_step(self, batch, batch_idx):
        ct0, ct1, self.segs, index_paths, self.cts, self.all_cts, self.all_segs = batch #self.cts, self.all_cts 為了存CT
        self.marked = [_[0] for _ in index_paths]
        self.Flow.eval()
        with torch.no_grad():
            # forward
            forward_flow = self.Flow(ct0, ct1)[-1]
            # backward
            backward_flow = self.Flow(ct1, ct0)[-1]

        if batch_idx == 0:
            self.forward_flows = forward_flow
            self.backward_flows = backward_flow
        else:
            self.forward_flows = torch.cat([self.forward_flows, forward_flow], dim=0)
            self.backward_flows = torch.cat([self.backward_flows, backward_flow], dim=0)

        del forward_flow
        del backward_flow
    def on_test_end(self):

        self.segs = [_[0] for _ in self.segs]
        self.cts  = [_[0] for _ in self.cts] #為了存CT

        # initialize saving dictionary
        # ct, None, None, None, ..., ct, None, None, None, ..., ct
        segs = {}
        cts = {} #為了存CT
        gen_f_cts = {} #為了存CT
        gen_b_cts = {} #為了存CT
        j = 0
        for i in range(len(self.forward_flows)+1):
            if i in self.marked:
                segs[i] = self.segs[j]
                cts[i] = self.cts[j] #為了存CT
                gen_f_cts[i] = self.cts[j] #為了存CT
                gen_b_cts[i] = self.cts[j] #為了存CT
                j += 1
            else:
                segs[i] = None
                cts[i] = None #為了存CT

        # forward
        for idx in range(len(self.forward_flows)):
            if idx in self.marked:
                f_ct = cts[idx].unsqueeze(0).unsqueeze(0) #為了存CT
                f_seg = segs[idx].unsqueeze(0)
            else:
                # forward
                f_ct = tools.warp(f_ct, self.forward_flows[idx-1]) #為了存CT
                f_seg = tools.warp_seg(f_seg, self.forward_flows[idx-1])
                segs[idx] = [f_seg]
                cts[idx] = [f_ct] #為了存CT

                gen_f_ct = tools.warp(self.all_cts[idx-1].unsqueeze(0), self.forward_flows[idx-1]) #為了存CT
                gen_f_cts[idx] = gen_f_ct[0, 0] #為了存CT
                
        # backward
        for idx in range(len(self.backward_flows), 0, -1):
            if idx in self.marked:
                b_ct = cts[idx].unsqueeze(0).unsqueeze(0) #為了存CT
                b_seg = segs[idx].unsqueeze(0)
            else:
                # backward
                b_ct = tools.warp(b_ct, self.backward_flows[idx]) #為了存CT
                b_seg = tools.warp_seg(b_seg, self.backward_flows[idx])
                segs[idx].append(b_seg)
                cts[idx].append(b_ct) #為了存CT

                gen_b_ct = tools.warp(self.all_cts[idx+1].unsqueeze(0), self.backward_flows[idx]) #為了存CT
                gen_b_cts[idx] = gen_b_ct[0, 0] #為了存CT

        shutil.rmtree(self.cfg['save_folder'], ignore_errors=True)
        forward_save_folder = os.path.join(self.cfg['save_folder'], 'forward')
        backward_save_folder = os.path.join(self.cfg['save_folder'], 'backward')
        union_save_folder = os.path.join(self.cfg['save_folder'], 'union')
        union_show_save_folder = os.path.join(self.cfg['save_folder'], 'union_show')
        combined_save_folder = os.path.join(self.cfg['save_folder'], 'combined')
        refined_save_folder = os.path.join(self.cfg['save_folder'], 'refined')

        ct_forward_save_folder = os.path.join(self.cfg['save_folder'], 'ct_forward') #為了存CT
        ct_backward_save_folder = os.path.join(self.cfg['save_folder'], 'ct_backward') #為了存CT
        gen_f_ct_save_folder = os.path.join(self.cfg['save_folder'], 'gen_f_ct') #為了存CT
        gen_b_ct_save_folder = os.path.join(self.cfg['save_folder'], 'gen_b_ct') #為了存CT
        os.makedirs(gen_f_ct_save_folder) #為了存CT
        os.makedirs(gen_b_ct_save_folder) #為了存CT
        os.makedirs(forward_save_folder)
        os.makedirs(backward_save_folder)
        os.makedirs(union_save_folder)
        os.makedirs(union_show_save_folder)
        os.makedirs(combined_save_folder)
        os.makedirs(refined_save_folder)
        os.makedirs(ct_forward_save_folder) #為了存CT
        os.makedirs(ct_backward_save_folder) #為了存CT

        for k, v in segs.items():
            if type(v) is list: # first of list is forward, second is backward
                fv = v[0][0, 0].cpu().numpy()
                np.save(os.path.join(forward_save_folder, 'seg{:03d}.npy'.format(k)), fv)
                bv = v[1][0, 0].cpu().numpy()
                np.save(os.path.join(backward_save_folder, 'seg{:03d}.npy'.format(k)), bv)
            else:
                v = v[0].cpu().numpy()
                np.save(os.path.join(forward_save_folder, 'seg{:03d}*.npy'.format(k)), v)
                np.save(os.path.join(backward_save_folder, 'seg{:03d}*.npy'.format(k)), v)
        
        for k, v in cts.items(): #為了存CT
            if type(v) is list: # first of list is forward, second is backward
                
                fv = v[0][0, 0].cpu().numpy() #(512, 512)

                np.save(os.path.join(ct_forward_save_folder, 'ct{:03d}.npy'.format(k)), fv)
              
                bv = v[1][0, 0].cpu().numpy()
                np.save(os.path.join(ct_backward_save_folder, 'ct{:03d}.npy'.format(k)), bv)
            else:
                v = v.cpu().numpy()
                np.save(os.path.join(ct_forward_save_folder, 'ct{:03d}*.npy'.format(k)), v)
                np.save(os.path.join(ct_backward_save_folder, 'ct{:03d}*.npy'.format(k)), v)
        
        for i in range(len(gen_f_cts)):
            np.save(os.path.join(gen_f_ct_save_folder, 'ct{:03d}*.npy'.format(i)), gen_f_cts[i].cpu().numpy()) #為了存CT
            np.save(os.path.join(gen_b_ct_save_folder, 'ct{:03d}*.npy'.format(i)), gen_b_cts[i].cpu().numpy()) #為了存CT

        keys = list(segs.keys())
        keys.sort()
        j = 0
        for k in keys[:-1]:
            if k in self.marked: #GT
                start = self.marked[j]
                end = self.marked[j+1]
                length = end - start + 1
                np.save(os.path.join(union_save_folder, 'seg_union{:03d}*.npy'.format(start)), self.segs[j][0].cpu().numpy())
                np.save(os.path.join(union_show_save_folder, 'seg_union_show{:03d}*.npy'.format(start)), self.segs[j][0].cpu().numpy())
                np.save(os.path.join(combined_save_folder, 'combined{:03d}*.npy'.format(start)), self.segs[j][0].cpu().numpy())
                np.save(os.path.join(refined_save_folder, 'refined{:03d}*.npy'.format(start)), self.segs[j][0].cpu().numpy())
                
                np.save(os.path.join(union_save_folder, 'seg_union{:03d}*.npy'.format(end)), self.segs[j+1][0].cpu().numpy())
                np.save(os.path.join(union_show_save_folder, 'seg_union_show{:03d}*.npy'.format(end)), self.segs[j+1][0].cpu().numpy())
                np.save(os.path.join(combined_save_folder, 'combined{:03d}*.npy'.format(end)), self.segs[j+1][0].cpu().numpy())
                np.save(os.path.join(refined_save_folder, 'refined{:03d}*.npy'.format(end)), self.segs[j+1][0].cpu().numpy())
                j += 1
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              
                ratio = (k - start) / (length - 1)
                ratio = torch.ones([1, 1, 1]).to(device) * ratio
                
                lengthInPatient = len(segs)
                global_ratio = (k / lengthInPatient)
                global_ratio = torch.ones([1, 1, 1]).to(device) * global_ratio
                
                
                f, b = segs[k]
                ratio, global_ratio = ratio.to(device), global_ratio.to(device)
                ct_t = self.all_cts[k][0].unsqueeze(0).unsqueeze(0)
                # input_corr = torch.cat([f, b, ct_t], dim=1)
                # seg_corr, forward_ct_region, backward_ct_region = self.correlationModel(input_corr)
                # seg_corr= torch.where(seg_corr >= 0.5, 1, 0)
                # np.save(os.path.join(combined_save_folder, 'combined{:03d}.npy'.format(k)), seg_corr[0, 0].cpu().numpy())

                # seg_union = f * (1 - ratio) + b * ratio
                # seg_union_show = torch.where(seg_union >= 0.5, 1, 0)
                # np.save(os.path.join(union_save_folder, 'seg_union{:03d}.npy'.format(k)), seg_union[0, 0].cpu().numpy())
                # np.save(os.path.join(union_show_save_folder, 'seg_union_show{:03d}.npy'.format(k)), seg_union_show[0, 0].cpu().numpy())
                # pos_ecd = self.PE(ratio, mode='test').unsqueeze(0).to(device) #position encoding

                seg_union = torch.where(f + b >= 1, 1, 0)
                
                # ct_union = seg_union * self.all_cts[k][0]
                # pos_ecd = self.PE(ratio, mode='test').unsqueeze(0).type_as(ct_union) #position encoding
                # global_pos_ecd = self.PE(global_ratio, mode='test').unsqueeze(0).type_as(ct_union) #position encoding
                # input_combined = torch.cat([f, b, pos_ecd, global_pos_ecd], dim=1)

                # seg_combined, forward_weight, backward_weight = self.combinedModel(input_combined)
                print('ratio', ratio.shape, ratio)

                pos_ecd = self.PE(ratio, mode='test').unsqueeze(0).to(device)  #position encoding
                global_pos_ecd = self.PE(global_ratio, mode='test').unsqueeze(0).to(device)  #position encoding
                input_combined = torch.cat([f, b, pos_ecd, global_pos_ecd], dim=1)
                seg_combined, forward_weight, backward_weight = self.combinedModel(input_combined)

                
                corr_volume = self.local_correlation_volume(ct_t, ct_t, self.cfg['radius'])
                input_DF = torch.cat([seg_combined, corr_volume], dim=1)
                
                filter = self.DynFilter_model(input_DF)
                residual = self.filtering(seg_combined, filter)
                print('residual', residual.max(), residual.min())
                print('residual', residual)
                np.save('resi', residual[0, 0].cpu().numpy())
                raise                
                seg_refined = seg_combined + residual
                
                # input_DF = torch.cat([seg_combined, ct_union], dim=1)
                # filter = self.DynFilter_model(input_DF)
                # residual = self.filtering(seg_combined, filter)
                # seg_refined = seg_combined + residual
                #算分要用過threshold的mask
                seg_combined= torch.where(seg_combined >= 0.5, 1, 0)
                # seg_refined = torch.where(seg_refined >= 0.5, 1, 0)
                np.save(os.path.join(combined_save_folder, 'combined{:03d}.npy'.format(k)), seg_combined[0, 0].cpu().numpy())
                # np.save(os.path.join(refined_save_folder, 'refined{:03d}.npy'.format(k)), seg_refined[0, 0].cpu().numpy())

    def configure_optimizers(self):
        opt = optim.Adam(self.combinedModel.parameters(), lr=self.cfg['lr'])
        # opt = optim.Adam(self.correlationModel.parameters(), lr=self.cfg['lr'])
        sch = optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.9)
        # sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.001, steps_per_epoch=1000, epochs=self.cfg['epochs'])
        return[opt], [sch]
    
    def forward(self, input):
        return self.combinedModel(input)

class CorrRefineModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.combinedModel = CombinedModel.load_from_checkpoint(cfg['combined_ckpt_path'], strict=False)
        self.combinedModel.eval()
        self.combinedModel.freeze()
        self.DynFilter_model = weight_combined.DFNet(cfg)
        self.filtering = weight_combined.dynFilter()

    def local_correlation_volume(self, A, B, radius):
        A_flat = A.view(A.size(0), A.size(1), -1)  # [batch_size, channels, height * width]
        B_unfolded = F.unfold(B, kernel_size=(2 * radius + 1), padding=radius)
        correlation = A_flat * B_unfolded
        correlation = correlation.view(A.size(0), (2 * radius + 1)**2, A.size(2), A.size(3))

        return correlation  
    
    def training_step(self, batch, batch_idx):
        self.combinedModel.Flow.freeze()
        forward_cts, backward_cts, seg0, seg_target, seg25, ratio, global_ratio, ct_t = batch
        # forward, configure batch
        seg_forward = seg0
        seg_forward = self.combinedModel.configure_batch(seg_forward, forward_cts)
        
        # backward, configure batch
        seg_backward = seg25
        backward_cts = list(reversed(backward_cts))
        seg_backward = self.combinedModel.configure_batch(seg_backward, backward_cts)
        seg_union = torch.where(seg_forward + seg_backward >= 1, 1, 0)
        # ct_union = seg_union * ct_t
        pos_ecd = self.combinedModel.PE(ratio, mode='train').type_as(seg_forward) #position encoding
        global_pos_ecd = self.combinedModel.PE(global_ratio, mode='train').type_as(seg_forward) #position encoding
        input_combined = torch.cat([seg_forward, seg_backward, pos_ecd, global_pos_ecd], dim=1)
        seg_combined, forward_weight, backward_weight = self.combinedModel(input_combined)
        
        corr_volume = self.local_correlation_volume(ct_t, ct_t, self.cfg['radius'])
        input_DF = torch.cat([seg_combined, corr_volume], dim=1)
        
        filter = self.DynFilter_model(input_DF)
        residual = self.filtering(seg_combined, filter)
        
        seg_refined = seg_combined + residual

        dice_loss_refined = self.combinedModel.Diceloss(seg_refined, seg_target)
        loss = dice_loss_refined
        # L1_loss_combined = self.L1loss(seg_combined, seg_target)
        # L1_loss_refined = self.L1loss(seg_refined, seg_target)

        # loss = L1_loss_combined + L1_loss_refined
        # Log
        # seg_combined  = torch.where(seg_combined >= 0.5, 1, 0)
        seg_refined = torch.where(seg_refined >= 0.5, 1, 0)
        residual = (residual - residual.min()) / (residual.max() - residual.min())
        self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.logger.experiment.add_scalar('Train/Dice_loss_refined', dice_loss_refined, self.global_step)
        if batch_idx % self.cfg['show_step'] == 0:
            self.logger.experiment.add_image('Train/seg0', seg0[0], self.global_step)
            self.logger.experiment.add_image('Train/seg25', seg25[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_target', seg_target[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_forward', seg_forward[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_backward', seg_backward[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_combined', seg_combined[0], self.global_step)
            self.logger.experiment.add_image('Train/seg_refined', seg_refined[0], self.global_step)
            # self.logger.experiment.add_image('Train/ct_union', ct_union[0], self.global_step)
            self.logger.experiment.add_image('Train/ct_t', ct_t[0], self.global_step)
            self.logger.experiment.add_image('Train/residual', residual[0], self.global_step)
            self.logger.experiment.add_image('Train/forward_weight', forward_weight[0], self.global_step)
            self.logger.experiment.add_image('Train/backward_weight', backward_weight[0], self.global_step)

        return loss
    
    def configure_optimizers(self):
        opt = optim.Adam(self.DynFilter_model.parameters(), lr=self.cfg['lr'])
        sch = optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.9)
        return[opt], [sch]
        


class ModelFactory:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        self.tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=cfg['model_name'],
            name='lightning_logs'
        )

        if args.task == 'flow':
            # Dataset
            if args.train or args.fast_run or args.resume:
                Dataset = FlowDataset(cfg, list_path=cfg['list_path'], train=True)
                self.loader = DataLoader(dataset=Dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
                Dataset = FlowDataset(cfg, list_path=cfg['val_list_path'], train=False)
                self.val_loader = DataLoader(dataset=Dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
            elif args.test:
                Dataset = FlowDataset(cfg, list_path=cfg['test_list_path'], train=False)
                self.loader = DataLoader(dataset=Dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])
            # Model
            self.model = LitModel(cfg)
        
        elif args.task == 'fusion_org':
            # Dataset
            if args.train or args.fast_run or args.resume:
                Dataset = FusionDataset(cfg, list_path=cfg['list_path'], train=True)
                self.loader = DataLoader(dataset=Dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])
            elif args.test:
                Dataset = Fusion_test_dataset(patient=args.patient, cfg=cfg, list_path=cfg['test_list_path'], train=False)
                self.loader = DataLoader(dataset=Dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
            elif args.test_auto:
                Dataset = Fusion_test_dataset(patient=args.patient, cfg=cfg, list_path=args.list_path, label=args.label, train=False)
                self.loader = DataLoader(dataset=Dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
                # 算時間
                # with open('Auto_InferTime_list.txt', 'a') as f:
                #     f.write('Label ' + str(args.label) + ' ' + args.patient + ' ')
            # Model
            self.model = DynFilterModel(cfg)
            
        elif args.task == 'combine':
            # Dataset
            if args.train or args.fast_run or args.resume:
                Dataset = FusionDataset(cfg, list_path=cfg['list_path'], train=True)
                self.loader = DataLoader(dataset=Dataset, batch_size=1, shuffle=True, num_workers=cfg['num_workers'])
            elif args.test:
                Dataset = Fusion_test_dataset(patient=args.patient, cfg=cfg, list_path=cfg['test_list_path'], train=False)
                self.loader = DataLoader(dataset=Dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
            self.model = CombinedModel(cfg)
        elif args.task == 'corr_refine':
            # Dataset
            if args.train or args.fast_run or args.resume:
                Dataset = FusionDataset1Organ(cfg, list_path=cfg['list_path'], train=True)
                self.loader = DataLoader(dataset=Dataset, batch_size=1, shuffle=True, num_workers=cfg['num_workers'])
            elif args.test:
                Dataset = Fusion_test_dataset(patient=args.patient, cfg=cfg, list_path=cfg['test_list_path'], train=False)
                self.loader = DataLoader(dataset=Dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
            self.model = CorrRefineModel(cfg)

    def train(self):
        trainer = pl.Trainer(max_epochs=self.cfg['epochs'], check_val_every_n_epoch=1,
                             logger=self.tb_logger, log_every_n_steps=5,
                             strategy='ddp_find_unused_parameters_true')
        if self.args.task == 'flow':
            #印出模型總參數量以及可訓練參數量以及summary
            # print('Total parameters:', sum(p.numel() for p in self.model.parameters()))
            # print('Trainable parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            # print(self.model)
            trainer.fit(model=self.model, train_dataloaders=self.loader, val_dataloaders=self.val_loader)
        elif self.args.task == 'fusion_org':
            trainer.fit(model=self.model, train_dataloaders=self.loader)
            
        elif self.args.task == 'combine':
            trainer.fit(model=self.model, train_dataloaders=self.loader)
        elif self.args.task == 'corr_refine':
            trainer.fit(model=self.model, train_dataloaders=self.loader)

    def resume(self):
        trainer = pl.Trainer(max_epochs=self.cfg['epochs'], check_val_every_n_epoch=1,
                             logger=self.tb_logger, log_every_n_steps=5,
                             strategy='ddp_find_unused_parameters_true')
        trainer.fit(model=self.model, train_dataloaders=self.loader, ckpt_path=self.args.ckpt_path)
        
    def fast_run(self):
        trainer = pl.Trainer(fast_dev_run=True)
        trainer.fit(model=self.model, train_dataloaders=self.loader)

    def test(self):
        trainer = pl.Trainer(devices=1, logger=self.tb_logger)
        #印出模型總參數量以及可訓練參數量以及summary
        # print('Total parameters:', sum(p.numel() for p in self.model.parameters()))
        # print('Trainable parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        # print(self.model)
        
        trainer.test(model=self.model, dataloaders=self.loader, ckpt_path=self.args.ckpt_path)
    def test_auto(self):
        trainer = pl.Trainer(devices=1, logger=self.tb_logger)
        trainer.test(model=self.model, dataloaders=self.loader, ckpt_path=self.args.ckpt_path)
        
def main():
    args = parser()
    with open(args.yaml_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    factory = ModelFactory(args, cfg)
    if args.fast_run:
        factory.fast_run()
    elif args.train:
        factory.train()
    elif args.resume:
        factory.resume()
    elif args.test:
        factory.test()
    elif args.test_auto:
        factory.test_auto()

if __name__ == '__main__':
    main()