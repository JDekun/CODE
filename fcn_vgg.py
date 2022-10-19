import math
import torch 
import torch.nn as nn
import torch.nn.init as init
# 建立block
class Block(nn.Module):

  def __init__(self, in_ch, out_ch):
    super(Block, self).__init__()
    self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_ch)
    self.relu1 = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.relu1(self.bn1(self.conv1(x)))
    return out
    
# 建立Layer
def make_layers(in_channels, layer_list):
  layers = []
  for v in layer_list:
    layers += [Block(in_channels, v)]
    in_channels = v          # 下次输入为上次输出
  return nn.Sequential(*layers)

class Layer(nn.Module):

  def __init__(self, in_channels, layer_list):
    super(Layer, self).__init__()
    self.layer = make_layers(in_channels, layer_list)
  
  def forward(self, x):
    out = self.layer(x)
    return out
    
# VGG 19 
# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
class VGG(nn.Module):
    def __init__(self):
      super(VGG, self).__init__()
      self.layer1 = Layer(3, [64, 64])
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.layer2 = Layer(64, [128, 128])
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.layer3 = Layer(128, [256, 256, 256, 256])
      self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.layer4 = Layer(256, [512, 512, 512, 512])
      self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.layer5 = Layer(512, [512, 512, 512, 512])
      self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
      f1 = self.pool1(self.layer1(x))
      f2 = self.pool2(self.layer2(f1))
      f3 = self.pool3(self.layer3(f2))
      f4 = self.pool4(self.layer4(f3))
      f5 = self.pool5(self.layer5(f4))
      return [f3, f4, f5]
  
# 建立上采样模块
class MergeUpsample(nn.Module):

    def __init__(self, in_cha1, in_cha2, out_chan):
      super(MergeUpsample, self).__init__()
      self.conv11 = Block(in_chan1, out_ch)
      self.conv12 = Block(in_chan2, out_ch)
      self.conv2 = Block(out_ch, out_ch)
      self.upsample = nn.ConvTranspose2d(out_chan1, 
          out_chan2 ,2 ,stride=2) 

    def forward(self, x, y):
        p1 = self.conv11(self.upsample(x))
        p2 = self.conv12(y)
        out = self.conv2(p1+p2) 
        return out

class FCNDecode(nn.Module):
    def __init__(self, n, in_channels, out_channels, upsample_ratio):
        super(FCNDecode, self).__init__()
        self.conv1 = Layer(in_channels, [out_channels]*n)
        self.trans_conv1 = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                upsample_ratio,
                stride=upsample_ratio)        
    def forward(self, x):
        out = self.trans_conv1(self.conv1(x))
        return out

# 建立FCN_Seg模型
class FCNSeg(nn.Module):
    def __init__(self, n, in_channels, out_channels, upsample_ratio):
        super(FCNSeg, self).__init__()
        self.encode = VGG()
        self.decode = FCNDecode(n, in_channels, out_channels, upsample_ratio)
        self.classifier = nn.Conv2d(out_channels, 10, 3, padding=1)
    def forward(self, x):
        feature_list = self.encode(x)
        out = self.decode(feature_list[-1])
        pro = self.classifier(out)
        return out

x = torch.randn((10, 3, 256, 256)) # batchsize, channel, h, w
model = FCNSeg(4, 512, 256, 32) # 卷积层， 输入通道， 输出通道，上采32
model.eval()
y = model(x)
y.size()  #torch.Size([10, 256, 256, 256])
