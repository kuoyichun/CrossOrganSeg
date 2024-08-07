import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import einops

class Pos_weights(nn.Module):
    def __init__(self, candidate_num):
        super(Pos_weights, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 128 * 128, 64)
        self.fc2 = nn.Linear(64, candidate_num) #之前寫成candidate_num=2，以為要輸出兩個權重值
        self.softmax = nn.Softmax(dim=1) #跟上一行一起寫錯了
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 128 * 128)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)        
        # x = self.sigmoid(x)
        
        return x


class dynFilter(nn.Module):
    def __init__(self, kernel_size=5, padding=2):
        super(dynFilter, self).__init__()

        self.padding = padding
        
        self.filter_localexpand = nn.Parameter(torch.reshape(
            torch.eye(kernel_size**2),
            (kernel_size**2, 1, kernel_size, kernel_size)
        ), requires_grad=False)
        

    def forward(self, x, filter):

        x_localexpand = []
        for c in range(x.size(1)):     
            x_localexpand.append(F.conv2d(x[:, c:c + 1, :, :], self.filter_localexpand, padding=self.padding))  # 5x5 kernel --> padding = 2

        x_localexpand = torch.cat(x_localexpand, dim=1)

        y = torch.sum(torch.mul(x_localexpand, filter), dim=1).unsqueeze(1)
        y = torch.tanh(y)
        
        return y #output residual

class Make_Dense(nn.Module):
    def __init__(self,intInput,intOutput):
        super(Make_Dense,self).__init__()

        self.dense_layer = nn.Sequential(
            # torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1), padding=dilation, dilation=dilation) #dilated convolution
            torch.nn.Conv2d(in_channels= intInput, out_channels= intOutput, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.ReLU()
        )

    def forward(self, input):
        Feature_d_output = self.dense_layer(input)
        return torch.cat((input, Feature_d_output), dim =1)


class RDB(nn.Module):
    def __init__(self, fusion_Feat, Conv_Num, Growth_rate):
        super(RDB, self).__init__()

        def LFF(intInput, intOutput):
            return torch.nn.Conv2d(in_channels= intInput, out_channels= intOutput, kernel_size=(1,1), stride = (1,1), padding= (0,0)) # Conv_1x1 layer

        self.Conv_Num = Conv_Num        # The number of Convolution layer before LFF

        Modules = [] # Sub_layer list #

        for i in range(Conv_Num):
            Modules.append(Make_Dense(fusion_Feat + i * Growth_rate, Growth_rate)) # Append dense layer

        self.Layer = nn.Sequential(*Modules) # Contiguous memory Mechanism

        self.local_feature_fusion = LFF(fusion_Feat + Conv_Num * Growth_rate, fusion_Feat) # G_0 : fusion_Feat, G : Growth_rate

    def forward(self, input):
        Feature_d_LF = self.local_feature_fusion(self.Layer(input))
        Feature_d = Feature_d_LF + input
        return Feature_d

#DFNet(16,4,16,6)
class DFNet(nn.Module):
    def __init__(self, cfg):
        """
        You are able to manipulate hyper parameters of RDN but,
        not able to load the ckpt model which has different hyper parameters.
        Arguments:
            fusion_Feat (int) : The number of channel of convolution layer.
            RDB_num     (int) : The number of Residual Dense Block.
            Growth_rate (int) : Growth rate of the number of convolution channel at dense layer.
            Conv_Num    (int) : The number of Convolution layers in Residual Dense Block.
        """
        super(DFNet, self).__init__()
        
        
        def SFE(intInput, intOutput):
            return torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1))

        def GFF(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(1, 1), stride=(1, 1),
                                padding=(0, 0)),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )
        
        self.fusion_Feat = cfg['fusion_Feat']
        self.RDB_Num = cfg['RDB_Num']
        self.Growth_rate = cfg['Growth_rate']
        self.candidate_num = cfg['candidate_num']

        self.SFE_First = SFE(cfg['candidate_num'], self.fusion_Feat) 
        self.SFE_Second = SFE(self.fusion_Feat, self.fusion_Feat)
        # self.SFE_weight = SFE(1, self.fusion_Feat) 
        self.pos_weights = Pos_weights(self.candidate_num)
        Modules = []

        for i in range(self.RDB_Num):
            Modules.append(RDB(fusion_Feat=self.fusion_Feat, Conv_Num=cfg['Conv_Num'], Growth_rate=self.Growth_rate))

        self.GRL = nn.Sequential(*Modules)
        self.Global_feature_fusion = GFF(self.RDB_Num * self.fusion_Feat, self.fusion_Feat)
        out_channel_num = 5 ** 2 * self.candidate_num
        self.Conv_Output = torch.nn.Conv2d(in_channels=self.fusion_Feat, out_channels=out_channel_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #有position encoding且最後用cat就要改in_channels=self.fusion_Feat*2
        #有position encoding但最後沒用cat用+就不用改 #240106 240104都這
    #input就是DFnet_input
        

    # def forward(self, input): #pos weight
    #     x = input[:,:-1] #[b, candidate_num, 512, 512] 
    #     pos = input[:,-1:] #[b, 1, 512, 512]
    #     weights = self.pos_weights(pos)
    #     # print('weights_sigmoid', weights)
        
    #     # print('weights', weights.shape)
    #     # # 原權重寫法
    #     # weights = einops.repeat(weights, "b c ->b c h w", h=512, w=512)
    #     # weighted = x * weights
    #     # 更改權重寫法231217
    #     forward_weights = weights
    #     forward_weights = einops.repeat(forward_weights, "b c ->b c h w", h=512, w=512)
    #     backward_weights = 1 - weights
        
    #     backward_weights = einops.repeat(backward_weights, "b c ->b c h w", h=512, w=512)
    #     weighted = torch.add(x[:,0] * forward_weights, x[:,1] * backward_weights)
        
    #     F_1 = self.SFE_First(x)
    #     F_2 = self.SFE_Second(F_1)  # [b, 16 ,512, 512]
    #     F_weight = self.SFE_First(weighted)  # [b, 16 ,512, 512] #240110之前的版本
    #     # F_weight = self.SFE_weight(weighted)
    #     F_all = F_2 + F_weight
        
    #     for i in range(self.RDB_Num): #4
    #         F_all = self.GRL[i](F_all)

    #         if i == 0:
    #             F_RDB_group = F_all

    #         else:
    #             F_RDB_group = torch.cat((F_RDB_group, F_all), dim=1)
    #     #print('F_RDB_group', F_RDB_group.shape) #[b, 64 ,512, 512]
        
    #     F_GF = self.Global_feature_fusion(F_RDB_group)  #[b, 16 ,512, 512] 
    #     # F_DF = torch.cat([F_GF, F_1], dim=1) #231227以前都把+寫成cat了[b, 32, 512, 512]
    #     F_DF = torch.add(F_GF, F_1) #把原始candidate的feature加回來 [b, 16, 512, 512]

    #     output = self.Conv_Output(F_DF) #[b, candidate_num*5*5, 512, 512]

    #     return output #生成filter
    
    #沒有position encoding 240104用這個
    def forward(self, input):
        input = input[:,:-1] #[b, candidate_num, 512, 512] 
        F_B1 = self.SFE_First(input)  # F_B1 == F_(-1)
        
        F_d = self.SFE_Second(F_B1)  # F_0 (d==0).
        
        for i in range(self.RDB_Num):
            F_d = self.GRL[i](F_d)

            if i == 0:
                F_RDB_group = F_d

            else:
                F_RDB_group = torch.cat((F_RDB_group, F_d), dim=1)

        
        F_GF = self.Global_feature_fusion(F_RDB_group)       
        F_DF = F_GF + F_B1
        output = self.Conv_Output(F_DF)
        return output 