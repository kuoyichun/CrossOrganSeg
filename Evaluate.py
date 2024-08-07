import glob, os
import tools
import torch
import numpy as np
import argparse
from matplotlib import pyplot as plt
from data import utils

def Calculate_patient_Dice(patient, label, dataset):
    forward_paths = glob.glob('/root/yichun/project/outputs/forward/*.npy')
    backward_paths = glob.glob('/root/yichun/project/outputs/backward/*.npy')
    union_paths = glob.glob('/root/yichun/project/outputs/union/*.npy')
    union_show_paths = glob.glob('/root/yichun/project/outputs/union_show/*.npy')
    residual_paths = glob.glob('/root/yichun/project/outputs/residual/*.npy')
    fused_paths = glob.glob('/root/yichun/project/outputs/fused/*.npy')
    refined_paths = glob.glob('/root/yichun/project/outputs/refined/*.npy')

    forward_paths.sort()
    backward_paths.sort()
    union_paths.sort()
    union_show_paths.sort()
    residual_paths.sort()
    fused_paths.sort()
    refined_paths.sort()

    if(dataset == 'CHAOS'):
        List = utils.open_txt('data/testList_CHAOS.txt')
 
    List = utils.open_txt('data/testList_label{}.txt'.format(label))
   
    gts = []
    cts = [] #為了存ct
    for p in List:
        if patient in p:
            cts.append(p) #為了存ct
            p = p.replace('CT', 'Seg')
            gts.append(p)
    gts.sort()

    return forward_paths, backward_paths, union_paths, union_show_paths, residual_paths, fused_paths, refined_paths, gts

def load_seg(path, label):
    seg = np.load(path)
    if label == 5:
        seg[256:, :] = 0
    elif label == 6:
        seg[:256, :] = 0
    seg = torch.tensor(seg).unsqueeze(0).unsqueeze(0)
    return seg

def evaluate_3D(forward_paths, backward_paths, union_paths, fused_paths, gts, label, patient, dataset):
    gt_3d = []
    for p in gts:
        seg = load_seg(p, label)
        if label == 5 or label == 6:
            seg = torch.where(seg == 2, 1, 0)
        else:
            seg = torch.where(seg == label, 1, 0)
        gt_3d.append(seg)
    gt_3d = torch.cat(gt_3d, dim=1)

    # forward
    forward_3d = []
    for p in forward_paths:
        seg = load_seg(p, label)
        forward_3d.append(seg)
    forward_3d = torch.cat(forward_3d, dim=1)    

    # backward
    backward_3d = []
    for p in backward_paths:
        seg = load_seg(p, label)
        backward_3d.append(seg)
    backward_3d = torch.cat(backward_3d, dim=1)

    # union_show
    union_show_3d = []
    for p in union_paths:
        seg = load_seg(p, label)
        union_show_3d.append(seg)
    union_show_3d = torch.cat(union_show_3d, dim=1)

    # fused / combined
    fused_3d = []
    for p in fused_paths:
        seg = load_seg(p, label)
        fused_3d.append(seg)
    fused_3d = torch.cat(fused_3d, dim=1)

    forward_dice = tools.dice(forward_3d, gt_3d)
    backward_dice = tools.dice(backward_3d, gt_3d)
    union_dice = tools.dice(union_show_3d, gt_3d)
    fused_dice = tools.dice(fused_3d, gt_3d)
    #用一個TXT檔存分數，格式為dataset, label, patient, forward_dice, backward_dice, union_dice, fused_dice
    with open('Auto_score_list.txt', 'a') as f:
        f.write(dataset + ' ' + str(label) + ' ' + patient + ' ' + str(round(forward_dice.item(), 4)) + ' ' + str(round(backward_dice.item(), 4)) + ' ' + str(round(union_dice.item(), 4)) + ' ' + str(round(fused_dice.item(), 4)) + '\n')

    # print('3D Dice score')
    # print("Forward Dice: ", round(forward_dice.item(), 4))
    # print("Backward Dice: ", round(backward_dice.item(), 4))
    # print("Union Dice: ", union_dice)
    # print("Fused Dice: ", fused_dice)
    # print(, round(forward_dice.item(), 4), round(backward_dice.item(), 4), round(union_dice.item(), 4), round(fused_dice.item(), 4))

#撰寫主程式main依序執行Calculate_patient_Dice和evaluate_3D
#使用parser輸入patient和label
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', type=str, required=True)
    parser.add_argument('--label', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    forward_paths, backward_paths, union_paths, union_show_paths, residual_paths, fused_paths, refined_paths, gts = Calculate_patient_Dice(args.patient, args.
    label, args.dataset)
    evaluate_3D(forward_paths, backward_paths, union_paths, fused_paths, gts, args.label, args.patient, args.dataset)



