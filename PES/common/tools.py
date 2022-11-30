import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def getTime():
    time_stamp = datetime.datetime.now()
    return time_stamp.strftime('%H:%M:%S')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, train_loader, optimizer, ceriation, epoch):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        logist = model(images)
        loss = ceriation(logist, labels)

        acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(0)
    return losses.avg, top1.avg.to("cpu", torch.float).item()


def evaluate(model, eva_loader, ceriation, prefix, ignore=-1):
    losses = AverageMeter('Loss', ':3.2f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(eva_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logist = model(images)

            loss = ceriation(logist, labels)
            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))

            losses.update(loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))

    if prefix != "":
        print(getTime(), prefix, round(top1.avg.item(), 2))

    return losses.avg, top1.avg.to("cpu", torch.float).item()


def evaluateWithBoth(model1, model2, eva_loader, prefix):
    top1 = AverageMeter('Acc@1', ':3.2f')
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(eva_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logist1 = model1(images)
            logist2 = model2(images)
            logist = (F.softmax(logist1, dim=1) + F.softmax(logist2, dim=1)) / 2
            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
            top1.update(acc1[0], images[0].size(0))

    if prefix != "":
        print(getTime(), prefix, round(top1.avg.item(), 2))

    return top1.avg.to("cpu", torch.float).item()


def predict(predict_loader, model):
    model.eval()
    preds = []
    probs = []

    with torch.no_grad():
        for images, _, _ in predict_loader:
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                logits = model(images)
                outputs = F.softmax(logits, dim=1)
                prob, pred = torch.max(outputs.data, 1)
                preds.append(pred)
                probs.append(prob)

    return torch.cat(preds, dim=0).cpu(), torch.cat(probs, dim=0).cpu()


def predict_softmax(predict_loader, model):

    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, images2, _ in predict_loader:
            if torch.cuda.is_available():
                images1 = Variable(images1).cuda()
                images2 = Variable(images2).cuda()
                logits1 = model(images1)
                logits2 = model(images2)
                outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
                softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()
