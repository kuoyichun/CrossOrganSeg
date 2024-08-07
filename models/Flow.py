import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('models')
from modules import *

class Model(nn.Module):
    def __init__(self, cfg): #flow dimension=2
        super().__init__()
        conv_dim = cfg['conv_dim']
        self.conv_layers = conv_layers = cfg['conv_layers']

        self.in_block = NetInBlock(2, conv_dim, 2)
        dim = conv_dim
        for i in range(1, conv_layers+1):
            p_dim = dim
            dim = p_dim * 2
            layer = NetDownBlock(p_dim, dim, 2)
            setattr(self, 'down_block{}'.format(i), layer)
        
        # Transformer on multi-scale
        self.TF_layers = TF_layers = cfg['TF_layers']
        dim = conv_dim
        res = cfg['resolution']
        for i in range(1, conv_layers+1):
            dim = dim * 2
            res = res // 2
            layer = [EncoderLayer(dim, res, res, k_size=3, pad_size=1) for _ in range(TF_layers)]
            setattr(self, 'transformer{}'.format(i), nn.Sequential(*layer))

        # upsample layer
        self.out_layers = out_layers = cfg['out_layers']
        for i in range(1, conv_layers+1):
            p_dim = dim if i == 1 else u_dim
            dim = dim // 2
            u_dim = (p_dim + dim) // 2
            layer = NetUpBlock(p_dim, dim, u_dim, 2)
            # setattr(self, 'up_block_01{}'.format(i), layer)
            setattr(self, 'up_block_10{}'.format(i), layer)
            
            for j in range(1, out_layers+1):
                op_dim = u_dim if j == 1 else o_dim
                o_dim = op_dim // 2
                if j == out_layers:
                    layer = NetOutSingleBlock(op_dim, 2)
                else:
                    layer = NetInBlock(op_dim, o_dim, 2)
                # setattr(self, 'out_01{}_{}'.format(i, j), layer)
                setattr(self, 'out_10{}_{}'.format(i, j), layer)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # residual layer
        #for i in range(1, conv_layers+1):
        #    layer = nn.ConvTranspose2d(2, 2, kernel_size=2, stride=2)
        #    setattr(self, 'residual_conv_{}'.format(i), layer)
        
    def forward(self, img0, img1):
        combined = torch.cat([img1, img0], dim=1)

        d0 = self.in_block(combined)
        ds =[]
        d = d0
        for i in range(1, self.conv_layers+1):
            layer = getattr(self, 'down_block{}'.format(i))
            d = layer(d)
            ds.append(d)

        fs = [d0]
        for i in range(1, self.conv_layers+1):
            layer = getattr(self, 'transformer{}'.format(i))
            d = layer(ds[i-1])
            fs.append(d)

        # flows01 = []
        # for i in range(1, self.conv_layers+1):
        #     layer = getattr(self, 'up_block_01{}'.format(i))
        #     if i == 1:
        #         d = layer(fs[-i], fs[-i-1])
        #     else:
        #         d = layer(d, fs[-i-1])

        #     out = d
        #     for j in range(1, self.out_layers+1):
        #         layer = getattr(self, 'out_01{}_{}'.format(i, j))
        #         out = layer(out)
        #     flows01.append(out)

        flows10 = []
        for i in range(1, self.conv_layers+1):
            layer = getattr(self, 'up_block_10{}'.format(i))
            if i == 1:
                d = layer(fs[-i], fs[-i-1])
            else:
                d = layer(d, fs[-i-1])
            #residual_layer = getattr(self, 'residual_conv_{}'.format(i))
            out = d
            for j in range(1, self.out_layers+1):
                layer = getattr(self, 'out_10{}_{}'.format(i, j))
                out = layer(out)
            # flows10.append(out)

            if i == 1:
               flows10.append(out)
            else:
               residual = self.upsample(p_out)
               flows10.append(torch.add(residual, out))
            p_out = out

        # return flows01, flows10
        return flows10

        
        
if __name__ == '__main__':
    input = torch.randn(1, 1, 512, 512)
    #after 1217:'conv_dim': 32
    #原本是:'conv_dim':16
    model = Model(cfg={'conv_dim': 16, 'conv_layers':4, 'TF_layers':8, 'out_layers': 3, 'resolution':512})
    model(input, input)

             
