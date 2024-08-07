import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    '''
    param:
        channel: channel of input feature
    input:
        x: feature
    return:
        out: channel attention feature
    '''
    def __init__(self, channel, ratio=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//ratio, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//ratio, channel, kernel_size=1, stride=1, padding=0),
        )
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out
        
        return self.act(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=True)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        
        return self.act(y)
    
class EncoderLayer(nn.Module):
    def __init__(self, C, H, W, k_size=3, pad_size=1):
        super().__init__()
        self.normC = nn.LayerNorm([C, H, W])
        self.normS = nn.LayerNorm([C, H, W])
        self.CA = ChannelAttention(C)
        self.SA = SpatialAttention()
        self.norm = nn.LayerNorm([C, H, W])
        self.ff = nn.Sequential(
            NetInBlock(C, C, 1, k_size=k_size, pad_size=pad_size),
            nn.ReLU(True),
            NetInBlock(C, C, 1, k_size=k_size, pad_size=pad_size)
        )
        
    def forward(self, x):

        Cout = self.normC(x)
        Cout = self.CA(Cout) * x
        Cout = Cout + x
        
        Sout = self.normS(x)
        Sout = self.SA(Sout) * x
        Sout = Sout + x
        
        out = Cout + Sout
        out = self.norm(out)
        y = self.ff(out)
        
        return y
    
class NetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, kernel_sz=3, pad = 1):
        '''
        ConvBlock = consistent convs
        for each conv, conv(5x5) -> BN -> activation(PReLU)
        params:
        in/out channels: output/input channels
        layers: number of convolution layers
        '''
        super(NetConvBlock, self).__init__()
        self.layers = layers
        self.afs = torch.nn.ModuleList() # activation functions
        self.convs = torch.nn.ModuleList() # convolutions
        self.bns = torch.nn.ModuleList()
        # first conv
        self.convs.append(nn.Conv2d( \
                in_channels, out_channels, kernel_size=kernel_sz, padding=pad))
        self.bns.append(nn.BatchNorm2d(out_channels))
        self.afs.append(nn.PReLU(out_channels))
        #self.afs.append(nn.ELU())
        for i in range(self.layers-1):
            self.convs.append(nn.Conv2d( \
                    out_channels, out_channels, kernel_size=kernel_sz, padding=pad))
            self.bns.append(nn.BatchNorm2d(out_channels))
            self.afs.append(nn.PReLU(out_channels))

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.convs[i](out)
            out = self.bns[i](out)
            out = self.afs[i](out)
        return out
    
class NetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(NetDownBlock, self).__init__()
        self.down = nn.Conv2d( \
                in_channels, out_channels, kernel_size=2, stride=2)
        self.af= nn.PReLU(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.convb = NetConvBlock(out_channels, out_channels, layers=layers)

    def forward(self, x):
        down = self.down(x)
        down = self.bn(down)
        down = self.af(down)
        out = self.convb(down)
        out = torch.add(out, down)
        return out
    
class NetUpBlock(nn.Module):
    def __init__(self, in_channels, br_channels, out_channels, layers):
        super(NetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        return out
    
class NetJustUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(NetJustUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels, out_channels, layers=layers)

    def forward(self, x):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = self.convb(up)
        #out = torch.add(out, up)
        return out

class NetInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1, k_size=3, pad_size=1):
        super(NetInBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.convb = NetConvBlock(in_channels, out_channels, layers=layers, kernel_sz=k_size, pad=pad_size)

    def forward(self, x):
        out = self.bn(x)
        out = self.convb(x)
        #out = torch.add(out, x)
        return out        
    
class NetOutSingleBlock(nn.Module):
    def __init__(self, in_channels, flows):
        super(NetOutSingleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, flows, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(flows)
        self.af_out = nn.PReLU(flows)
        #self.af_out = nn.PReLU(flows)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn_out(out)
        out = self.af_out(out)
        return out