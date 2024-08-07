import torch
import torchvision
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np
import random

import os

import sys
sys.path.append('data')
import utils

class BaseDataset(Dataset):
    def __init__(self):
        self.totensor = torchvision.transforms.ToTensor()
        self.angles = [0, 90, 180, 270]

    def normalize(self, x):
        if x.max() - x.min() == 0:
            print(self.path)
        return (x - x.min()) / (x.max() - x.min())
    
    def load_seg(self, path):
        seg = np.load(path)
        seg = np.where(seg == 2, 1, 0) #which label(2 for Abd train)
        seg = self.totensor(seg).type(torch.FloatTensor)

        return seg
    
class FlowDataset(BaseDataset):
    def __init__(self, cfg, list_path, train=True):
        super().__init__()
        self.train = train

        # open list
        self.List = utils.open_txt(list_path)

    def __getitem__(self, index):
        path = self.List[index]
        dirpath = os.path.dirname(path)
        self.path = path
        # ct0
        ct_path0 = path
        ct0 = np.load(ct_path0).astype('float32')
        ct0 = self.normalize(ct0)
        ct0 = self.totensor(ct0)

        # ct1
        number0 = int(os.path.basename(path).split('.')[0])
        number1 = number0 + 1
        ct_path1 = os.path.join(dirpath, '{:03d}.npy'.format(number1))
        if ct_path1 not in self.List:
            number1 = number0 - 1
            ct_path1 = os.path.join(dirpath, '{:03d}.npy'.format(number1))
        
        ct1 = np.load(ct_path1).astype('float32')
        ct1 = self.normalize(ct1)
        ct1 = self.totensor(ct1)

        return ct0, ct1
    
    def __len__(self):
        return len(self.List)
    
class FusionDataset1Organ(BaseDataset):
    def __init__(self, cfg, list_path, train=True):
        super().__init__()
        self.train = train

        # open list
        self.List = utils.open_txt(list_path)
        # patient length
        self.patient_dict = {}
        slices = []
        for path in self.List:
            patient = path.split('/')[-3]
            if slices == []:
                pre_patient = patient
            if patient == pre_patient:
                slices.append(path)
            elif patient != pre_patient:
                self.patient_dict[pre_patient] = slices
                slices = [path]
                pre_patient = patient
        self.patient_dict[pre_patient] = slices
        

    def __getitem__(self, index):
        path = self.List[index]
        patient = path.split('/')[-3]
        slices = self.patient_dict[patient]
        length = len(slices)
        quater_length = length // 8
        # ct0
        ct0_path = random.choice(slices[:-quater_length])
        dirpath = os.path.dirname(ct0_path)

        # ct0~ct25
        number0 = int(os.path.basename(ct0_path).split('.')[0])
        number25 = number0 + quater_length
        paths = [os.path.join(dirpath, '{:03d}.npy'.format(number)) for number in range(number0, number25+1)]

        # load ct
        cts = []
        for path in paths:
            ct = np.load(path).astype('float32')
            ct = self.normalize(ct)
            ct = self.totensor(ct)
            cts.append(ct)

        number_paths = len(paths)
        target_number = random.randint(1, number_paths-2)

        forward_cts = cts[:target_number+1]
        backward_cts = cts[target_number:]
        ct0_global_index = slices.index(ct0_path)
        global_ratio = torch.tensor((target_number + ct0_global_index )/ length)
        ratio = torch.tensor(target_number / (number_paths - 1))
        segs_path = [paths[0].replace('CT', 'Seg'), paths[target_number].replace('CT', 'Seg'), paths[-1].replace('CT', 'Seg')]

        # load segmentation
        seg0 = self.load_seg(segs_path[0])
        seg_target = self.load_seg(segs_path[1])
        seg25 = self.load_seg(segs_path[2])
        ct_t = cts[target_number]
        
        return forward_cts, backward_cts, seg0, seg_target, seg25, ratio, global_ratio, ct_t
    
    def __len__(self):
        #return 1
        return len(self.List) // 4
        # return 100


class Fusion_test_dataset(FusionDataset1Organ):
    def __init__(self, patient, cfg, list_path, train=True):
        super().__init__(cfg, list_path, train)
        self.slices = self.patient_dict[patient]
        self.length = len(self.slices)
        # 3 annotation
        # self.index_paths = index_paths = [0, int(self.length*0.5), self.length-1]
        # 5 annotation
        self.index_paths = index_paths = [0, int(self.length*0.25), int(self.length*0.5), int(self.length*0.75), self.length-1]
        # 7 annotation
        # self.index_paths = index_paths = [0, int(self.length/6*1), int(self.length/6*2), int(self.length/6*3), int(self.length/6*4), int(self.length/6*5), self.length-1]
        paths = [self.slices[index] for index in index_paths]
        seg_paths = [path.replace('CT', 'Seg') for path in paths]
        self.segs = [self.load_seg(path) for path in seg_paths]
        self.cts = [self.normalize(np.load(path).astype('float32')) for path in paths] #為了存CT
        self.all_cts = [self.normalize(np.load(path).astype('float32')) for path in self.slices] #為了存CT

    def __getitem__(self, index):
        # ct0
        ct0_path = self.slices[index]
        ct0 = np.load(ct0_path).astype('float32')
        ct0 = self.normalize(ct0)
        ct0 = self.totensor(ct0)
        # ct1
        ct1_path = self.slices[index+1]
        ct1 = np.load(ct1_path).astype('float32')
        ct1 = self.normalize(ct1)
        ct1 = self.totensor(ct1)
        
        
        return ct0, ct1, self.segs, self.index_paths, self.cts, self.all_cts #self.cts 為了存CT


    def __len__(self):
        return len(self.slices)-1

        



if __name__ == '__main__':
    import yaml
    with open('/root/yichun/project/configs/test.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # dataset = FlowDataset(cfg={'list_path': '/root/yichun/project/data/trainList.txt'})
    # dataset = FusionDataset(cfg=cfg, list_path='/root/yichun/project/data/trainList_fusion.txt')
    dataset = Fusion_test_dataset(patient='train_0036_0000', cfg=cfg, list_path='/root/yichun/project/data/testList_label1.txt')
    for i in dataset:
        print(i[2], i[3])
        break