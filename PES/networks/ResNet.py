import torch.nn as nn
import torch.nn.functional as F


class MySequential(nn.Sequential):
    def forward(self, x, adv):
        for module in self._modules.values():
            x = module(x, adv=adv)
        return x
# ResNet in PyTorch.
# BasicBlock and Bottleneck module is from the original ResNet paper:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# PreActBlock and PreActBottleneck module is from the later paper:
# [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Identity Mappings in Deep Residual Networks. arXiv:1603.05027


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(PreActBlock, self).__init__()
        self.bn_adv_momentum = bn_adv_momentum
        self.bn_adv_flag = bn_adv_flag
        self.bn1 = nn.BatchNorm2d(in_planes)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(in_planes, momentum=self.bn_adv_momentum)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x, adv=False):
        if adv and self.bn_adv_flag:
            out = F.relu(self.bn1_adv(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2_adv(out)))
            out += shortcut
        else:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(in_planes, momentum=self.bn_adv_momentum)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn3_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x, adv=False):
        if adv and self.bn_adv_flag:
            out = F.relu(self.bn1_adv(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2_adv(out)))
            out = self.conv3(F.relu(self.bn3_adv(out)))
            out += shortcut
        else:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out = self.conv3(F.relu(self.bn3(out)))
            out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,  bn_adv_flag=False, bn_adv_momentum=0.01):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.bn_adv_momentum = bn_adv_momentum
        self.bn_adv_flag = bn_adv_flag
        self.num_classes = num_classes
        self.block = block
        self.num_blocks = num_blocks

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        if bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(64, momentum = self.bn_adv_momentum)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, bn_adv_flag=False, bn_adv_momentum=0.01):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn_adv_flag=bn_adv_flag, bn_adv_momentum = bn_adv_momentum))
            self.in_planes = planes * block.expansion
        return MySequential(*layers)

    def forward(self, x, adv=False, return_features=False):
        out = x
        out = self.conv1(out)
        if adv and self.bn_adv_flag:
            out = self.bn1_adv(out)
        else:
            out = self.bn1(out)
        out = F.relu(out)

        out1 = self.layer1(out, adv=adv)
        out2 = self.layer2(out1, adv=adv)
        out3 = self.layer3(out2, adv=adv)
        out4 = self.layer4(out3, adv=adv)

        out = F.avg_pool2d(out4, 4)
        z = out.view(out.size(0), -1)
        out = self.linear(z)
        if return_features:
            return out, [out1, out2, out3, out4]
        else:
            return out#, nn.functional.normalize(z, dim=1)

    def renew_layers(self, last_num_layers):
        if last_num_layers >= 3:
            print("re-initalize block 2")
            self.in_planes = 64  # reset input dimension to 1th block output
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                           bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)

        if last_num_layers >= 2:
            print("re-initalize block 3")
            self.in_planes = 128  # reset input dimension to 2th block output
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                           bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        if last_num_layers >= 1:
            print("re-initalize block 4")
            self.in_planes = 256  # reset input dimension to 3th block output
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                           bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)

        print("re-initalize the final layer")
        self.linear = nn.Linear(512, self.num_classes)

    def update_num_layers(self, last_num_layers):
        return


def PreActResNet18(num_classes, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

