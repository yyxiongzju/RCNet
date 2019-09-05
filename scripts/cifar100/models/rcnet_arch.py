import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,int(C//g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class RCBlock3x3(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, exp_group=1, pro_group=1):
        super(RCBlock3x3, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, groups=exp_group, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, groups=pro_group, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shuffle1 = ShuffleBlock(groups=pro_group)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.shuffle1(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class RCNet(nn.Module):
    # (expansion, out_planes, num_blocks, stride, exp_group, pro_group)
    def __init__(self, cfg, scale=1.0, input_size=32, num_classes=100):
        super(RCNet, self).__init__()
        in_planes=32
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)

        layers = []
        for expansion, out_planes, num_blocks, stride, exp_group, pro_group in cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(RCBlock3x3(in_planes, out_planes, expansion, stride, exp_group, pro_group))
                in_planes = out_planes

        self.layers = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(in_planes, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




