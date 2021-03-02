import torch.nn as nn
import torch


def conv3x3(in_channels, out_channels, stride = 1,padding = 1):
    """3x3 convolution with Padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,bias=False,padding = padding)


def conv1x1(in_channels, out_channels, stride = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

#For Resnet 50( Imaprove on Performance BottleNeck Instead of Basic Residual)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,norm_layer = None,stride = 1,downsample = None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self,in_channels, out_channels, norm_layer,stride = 1,downsample = None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True) # A small amout of memory compare to inplace = False(default)
                                          # Only if there is no Error
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self,x):
        identity = x # For shortcut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
          identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
  def __init__(self,block,layers,num_classes = 1000, zero_init_residual = False,feat_dim=100):
    super(ResNet, self).__init__()
    self._norm_layer = nn.BatchNorm2d
    self.inplanes = 64
    
    #From RGB(3_channels -> 64 Channels) Resnet Paper Arch
    #7x7 Conv, 64,/2
    self.conv1 = nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding=3
                           ,bias=False)# Bias False as Already in Batch
    self.bn1 = nn.BatchNorm2d(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    
    #pool, /2
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    #Group of Residual Blocks
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    #layer Befor fully connected(max pool)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    ##original 
    #self.fc = nn.Linear(512, num_classes)
    #self.linear_label = nn.Linear(512 * block.expansion, num_classes, bias=False)
    self.linear_label = nn.Linear(512 * block.expansion, num_classes)
    #For our problem 
    # For creating the embedding to be passed into the Center Loss criterion
    #self.linear_closs = nn.Linear(512, feat_dim, bias=False)
    self.linear_closs = nn.Linear(512*block.expansion, feat_dim)
    #self.relu_closs = nn.ReLU(inplace=True)  # Test with  relu and without
    #self.relu_closs = nn.LeakyReLU(inplace=True)  #Maybe only for classification use relu

    #Initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
    
  def _make_layer(self, block, planes, blocks,stride=1):
    norm_layer = self._norm_layer
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes*block.expansion),
            )
    layers = []
    layers.append(block(self.inplanes, planes, norm_layer,stride,downsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes,norm_layer=norm_layer))
    return nn.Sequential(*layers)

  def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #output = x.detach().clone()
        label_output = self.linear_label(x)
        #label_output = label_output/torch.norm(self.linear_label.weight, dim=1)

        # Create the feature embedding for the Center Loss
        class_output = self.linear_closs(x)
        #class_output = self.relu_closs(class_output)

        return class_output, label_output

  def forward(self, x):
        return self._forward_impl(x)

def _resnet(block,layers,**kwargs):
    model = ResNet(block, layers,**kwargs)
    return model

def resnet18(**kwargs):
  return _resnet(BasicBlock,[2,2,2,2],**kwargs)

def resnet50(**kwargs):
    r"""ResNet-50 model
    num_classes=num_classes,feat_dim=100
    Args:
        num_classes (int): Number of Classes for output layer 
        feat_dim (int): Size of faceembedding(layer Obtained by deleting output layer)
    """
    return _resnet(Bottleneck, [3, 4, 6, 3],**kwargs)
