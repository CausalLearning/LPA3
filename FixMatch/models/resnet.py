import torch.nn as nn
import torch.nn.functional as F


class MySequential(nn.Sequential):
    def forward(self, x, adv):
        for module in self._modules.values():
            x = module(x, adv=adv)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(BasicBlock, self).__init__()
        self.bn_adv_momentum = bn_adv_momentum
        self.bn_adv_flag = bn_adv_flag
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)

        self.shortcut = nn.Sequential()
        self.shortcut_bn = None
        self.shortcut_bn_adv = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes)
            if self.bn_adv_flag:
                self.shortcut_bn_adv = nn.BatchNorm2d(self.expansion * planes, momentum=self.bn_adv_momentum)

    def forward(self, x, adv=False):
        if adv and self.bn_adv_flag:
            out = F.relu(self.bn1_adv(self.conv1(x)))
            out = self.conv2(out)
            out = self.bn2_adv(out)
            if self.shortcut_bn_adv:
                out += self.shortcut_bn_adv(self.shortcut(x))
            else:
                out += self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.conv2(out)
            out = self.bn2(out)
            if self.shortcut_bn:
                out += self.shortcut_bn(self.shortcut(x))
            else:
                out += self.shortcut(x)

        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(Bottleneck, self).__init__()
        self.bn_adv_momentum = bn_adv_momentum
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn_adv_flag = bn_adv_flag

        self.bn1 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        if self.bn_adv_flag:
            self.bn3_adv = nn.BatchNorm2d(self.expansion * planes, momentum=self.bn_adv_momentum)

        self.shortcut = nn.Sequential()
        self.shortcut_bn = None
        self.shortcut_bn_adv = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes)
            if self.bn_adv_flag:
                self.shortcut_bn_adv = nn.BatchNorm2d(self.expansion * planes, momentum=self.bn_adv_momentum)

    def forward(self, x, adv=False):

        if adv and self.bn_adv_flag:

            out = F.relu(self.bn1_adv(self.conv1(x)))
            out = F.relu(self.bn2_adv(self.conv2(out)))
            out = self.bn3_adv(self.conv3(out))
            if self.shortcut_bn_adv:
                out += self.shortcut_bn_adv(self.shortcut(x))
            else:
                out += self.shortcut(x)
        else:

            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.shortcut_bn:
                out += self.shortcut_bn(self.shortcut(x))
            else:
                out += self.shortcut(x)

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, num_classes=2, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.bn_adv_momentum = bn_adv_momentum
        self.bn_adv_flag = bn_adv_flag
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(64, momentum = self.bn_adv_momentum)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, bn_adv_flag=False, bn_adv_momentum=0.01):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn_adv_flag=bn_adv_flag, bn_adv_momentum = bn_adv_momentum))
            self.in_planes = planes * block.expansion
        return MySequential(*layers)

    def forward(self, x, return_feature=False, adv=False):
        if adv and self.bn_adv_flag:
            out = F.relu(self.bn1_adv(self.conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out1 = self.layer1(out, adv=adv)
        out2 = self.layer2(out1, adv=adv)
        out3 = self.layer3(out2, adv=adv)
        out4 = self.layer4(out3, adv=adv)

        out = self.avgpool(out4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if return_feature:
            return out, (out1, out2, out3, out4)
        else:
            return out


def ResNet18(num_classes, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)


def ResNet50(num_classes, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)