import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
logger = logging.getLogger(__name__)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class MySequential(nn.Sequential):
    def forward(self, x, adv):
        for module in self._modules.values():
            x = module(x, adv=adv)
        return x


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(BasicBlock, self).__init__()
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(in_planes,  momentum=self.bn_adv_momentum)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(out_planes, momentum=self.bn_adv_momentum)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x, adv=False):
        if adv and self.bn_adv_flag:
            if not self.equalInOut and self.activate_before_residual == True:
                x = self.relu1(self.bn1_adv(x))
            else:
                out = self.relu1(self.bn1_adv(x))
            out = self.relu2(self.bn2_adv(self.conv1(out if self.equalInOut else x)))
        else:
            if not self.equalInOut and self.activate_before_residual == True:
                x = self.relu1(self.bn1(x))
            else:
                out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual, bn_adv_flag=False, bn_adv_momentum=0.01):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum))
        return MySequential(*layers)

    def forward(self, x, adv):
        return self.layer(x, adv)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(WideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.layer1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        # 2nd block
        self.layer2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        # 3rd block
        self.layer3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        if bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(channels[3], momentum=self.bn_adv_momentum)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.linear = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]
        self.normalize = Normalize()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_feature=False, adv=False):
        out = self.conv1(x)
        out1 = self.layer1(out, adv)
        out2 = self.layer2(out1, adv)
        out3 = self.layer3(out2, adv)
        if adv and self.bn_adv_flag:
            out = self.relu(self.bn1_adv(out3))
        else:
            out = self.relu(self.bn1(out3))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        if return_feature:
            return self.linear(out), (out1, out2, out3)
        else:
            return self.linear(out)


class WideResNetVar(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(WideResNetVar, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor, 128 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.layer1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        # 2nd block
        self.layer2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        # 3rd block
        self.layer3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        self.layer4 = NetworkBlock(
            n, channels[3], channels[4], block, 2, drop_rate, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[4], momentum=0.001)
        if bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(channels[4], momentum=self.bn_adv_momentum)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.linear = nn.Linear(channels[4], num_classes)
        self.channels = channels[4]
        self.normalize = Normalize()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_feature=False, adv=False):
        out = self.conv1(x)
        out1 = self.layer1(out, adv)
        out2 = self.layer2(out1, adv)
        out3 = self.layer3(out2, adv)
        out4 = self.layer4(out3, adv)
        if adv and self.bn_adv_flag:
            out = self.relu(self.bn1_adv(out4))
        else:
            out = self.relu(self.bn1(out4))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        if return_feature:
            return self.linear(out), (out1, out2, out3, out4)
        else:
            return self.linear(out)


def build_wideresnet(depth, widen_factor, dropout, num_classes, bn_adv_flag=False, bn_adv_momentum=0.01):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return WideResNet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes,
                      bn_adv_flag=bn_adv_flag,
                      bn_adv_momentum=bn_adv_momentum)


def build_wideresnetVar(depth, widen_factor, dropout, num_classes, bn_adv_flag=False, bn_adv_momentum=0.01):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return WideResNetVar(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes,
                      bn_adv_flag=bn_adv_flag,
                      bn_adv_momentum=bn_adv_momentum)