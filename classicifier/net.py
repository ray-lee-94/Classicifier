import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
def conv3x3(in_plane, out_plane, stride=1):
    return nn.Conv2d(in_plane, out_plane, kernel_size=(3, 3), stride=stride, padding=1, bias=False)


def conv1x1(in_plane, out_plane, stride=1):
    return nn.Conv2d(in_plane, out_plane, kernel_size=(1, 1), stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_plane, out_plane, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_plane, out_plane, stride=1)
        self.bn1 = nn.BatchNorm2d(out_plane)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_plane, out_plane)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu1(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_plane, out_plane, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(in_plane, out_plane)
        self.bn1 = nn.BatchNorm2d(out_plane)
        self.conv2 = conv3x3(out_plane, out_plane, stride)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.conv3 = conv1x1(out_plane, out_plane * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_plane * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplane = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,num_classes,bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.__make_layer(block, 64, layers[0])
        self.layer2 = self.__make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.__make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.__make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_layer(self, block, out_plane, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplane != out_plane * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplane, out_plane * block.expansion, stride),
                                       nn.BatchNorm2d(out_plane * block.expansion), )
        layers = []
        layers.append(block(self.inplane, out_plane, stride, downsample))
        self.inplane = out_plane * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplane, out_plane))
        return nn.Sequential(*layers)
    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(self.bn1(x))
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        ip1=self.fc1(x)
        ip2=self.fc2(ip1)
        return ip1,F.log_softmax(ip2,dim=1)

    def resnet50(**kwargs):
        model=ResNet(Bottleneck,[3,4,6,3],**kwargs)
        return model
