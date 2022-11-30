import os
import os.path
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torchvision.datasets import CIFAR10, CIFAR100

from networks.ResNet import PreActResNet18
from common.tools import AverageMeter, getTime, evaluate, predict_softmax, train
from common.NoisyUtil import Train_Dataset, Semi_Labeled_Dataset, Semi_Unlabeled_Dataset, dataset_split
import wandb
from typing import List, Optional, Tuple, Union, cast
import pdb


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_path', type=str, default='../data', help='data directory')
parser.add_argument('--data_percent', default=1, type=float, help='data number percent')
parser.add_argument('--noise_type', default='symmetric', type=str)
parser.add_argument('--noise_rate', default=0.5, type=float, help='corruption rate, should be less than 1')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=5e-4)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--optim', default='cos', type=str, help='step, cos')

parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--lambda_u', default=150, type=float, help='weight for unsupervised loss')
parser.add_argument('--PES_lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--T1', default=0, type=int, help='if 0, set in below')
parser.add_argument('--T2', default=5, type=int, help='default 5')
# LPA3 parameters
parser.add_argument('--bound', default=0.02, type=float, help='bound for adversarial perturbation')
parser.add_argument('--num_iterations', default=5, type=int, help='fast lagrangian iterations')
parser.add_argument('--lam', default=1, type=float, help='fast lagrangian lambda')
args = parser.parse_args()
print(args)
os.system('nvidia-smi')

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
mu_cifar100 = torch.tensor(cifar100_mean).view(3,1,1)
std_cifar100 = torch.tensor(cifar100_std).view(3,1,1)
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
mu_cifar10 = torch.tensor(cifar10_mean).view(3,1,1)
std_cifar10 = torch.tensor(cifar10_std).view(3,1,1)


def reconst_images(x_adv, strong_x):
    grid_X = torchvision.utils.make_grid(strong_x[:10].data, nrow=10, padding=2, normalize=True)
    grid_AdvX = torchvision.utils.make_grid(x_adv[:10].data, nrow=10, padding=2, normalize=True)
    grid_Delta = torchvision.utils.make_grid(x_adv[:10]-strong_x[:10].data, nrow=10, padding=2, normalize=True)
    grid = torch.cat((grid_X, grid_AdvX, grid_Delta), dim=1)
    wandb.log({"Batch.jpg": [
        wandb.Image(grid)]}, commit=False)
    wandb.log({'His/l2_norm': wandb.Histogram((x_adv - strong_x).reshape(strong_x.shape[0], -1).norm(dim=1).cpu().detach().numpy(), num_bins=512),
               }, commit=False)


def normalize_flatten_features(
    features: Tuple[torch.Tensor, ...],
    eps=1e-10,
) -> torch.Tensor:

    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2] *
                                      feature_layer.size()[3])))
            .view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_attack(model, inputs, targets_u, y_ori, flat_feat_ori, args):
    if args.dataset == 'cifar10':
        upper_limit = ((1 - mu_cifar10) / std_cifar10).cuda()
        lower_limit = ((0 - mu_cifar10) / std_cifar10).cuda()
    elif args.dataset == 'cifar100':
        upper_limit = ((1 - mu_cifar100) / std_cifar100).cuda()
        lower_limit = ((0 - mu_cifar100) / std_cifar100).cuda()

    perturbations = torch.zeros_like(inputs)
    perturbations.uniform_(-0.01, 0.01)
    perturbations.data = clamp(perturbations, lower_limit - inputs, upper_limit - inputs)
    perturbations.requires_grad = True
    for attack_iter in range(args.num_iterations):
        # Decay step size, but increase lambda over time.
        step_size = \
            args.bound * 0.1 ** (attack_iter / args.num_iterations)
        lam = \
            args.lam * 0.1 ** (1 - attack_iter / args.num_iterations)

        if perturbations.grad is not None:
            perturbations.grad.data.zero_()

        inputs_adv = inputs + perturbations

        logits_adv, feat_adv = model(inputs_adv, adv=True, return_features=True)
        prob_adv = torch.softmax(logits_adv, dim=-1)
        y_adv = torch.log(torch.gather(prob_adv, 1, targets_u.view(-1, 1)).squeeze(dim=1))

        pip = (normalize_flatten_features(feat_adv) - \
        normalize_flatten_features(flat_feat_ori)).norm(dim=1).mean()
        constraint = y_ori - y_adv
        loss = -pip + lam * F.relu(constraint - args.bound).mean()
        loss.backward()

        grad = perturbations.grad.data
        grad_normed = grad / \
                      (grad.reshape(grad.size()[0], -1).norm(dim=1)
                       [:, None, None, None] + 1e-8)
        with torch.no_grad():
            y_after = torch.log(torch.gather(torch.softmax(
                         model(inputs + perturbations - grad_normed * 0.1, adv=True), dim=1),
                         1, targets_u.view(-1, 1)).squeeze(dim=1))
            dist_grads = torch.abs( y_adv - y_after
                         ) / 0.1
            norm = step_size / (dist_grads + 1e-4)
        perturbation_updates = -grad_normed * norm[:, None, None, None]

        perturbations.data = clamp(perturbations + perturbation_updates,
                                   lower_limit - inputs, upper_limit - inputs).detach()

    inputs_adv = (inputs + perturbations).detach()
    return inputs_adv


args.model_dir = 'model/'
if not os.path.exists(args.model_dir):
    os.system('mkdir -p %s' % (args.model_dir))

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn.deterministic = True
    cudnn.benchmark = True


def create_model(num_classes=10):
    model = PreActResNet18(num_classes, bn_adv_flag=True, bn_adv_momentum=0.01)
    model.cuda()
    return model


def linear_rampup(current, warm_up=20, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


# MixMatch Training
def MixMatch_train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader, class_weights):
    net.train()
    if epoch >= args.num_epochs/2:
        args.alpha = 0.75

    losses = AverageMeter('Loss', ':6.2f')
    losses_lx = AverageMeter('Loss_Lx', ':6.2f')
    losses_lu = AverageMeter('Loss_Lu', ':6.5f')

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = int(50000/args.batch_size)
    for batch_idx in range(num_iter):
        try:
            inputs_x, inputs_x2, targets_x = labeled_train_iter.next()
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x2, targets_x = labeled_train_iter.next()

        try:
            inputs_u, inputs_u2, targets_ux = unlabeled_train_iter.next()
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, targets_ux = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        targets_ux = targets_ux.cuda()
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1, 1), 1)
        inputs_x, inputs_x2, targets_x = inputs_x.cuda(), inputs_x2.cuda(), targets_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
            ptu = pu**(1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            #adv
            inputs_ori = torch.cat([inputs_u, inputs_u2], dim=0)
            targets_ori = torch.cat([targets_u, targets_u], dim=0)

            logits_ori, feat_ori = net(inputs_ori, adv=True, return_features=True)
            _, targets_uadv = torch.max(logits_ori, 1)
            prob = torch.softmax(logits_ori, dim=-1)
            y_w = torch.log(torch.gather(prob, 1, targets_uadv.view(-1, 1)).squeeze(dim=1))

        inputs_adv = get_attack(model, inputs_ori, targets_uadv, y_w, feat_ori, args)
        optimizer.zero_grad()

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        mixmatch_l = np.random.beta(args.alpha, args.alpha)
        mixmatch_l = max(mixmatch_l, 1 - mixmatch_l)
        # adv mixmatch
        all_inputsadv = torch.cat([inputs_x, inputs_x2, inputs_adv], dim=0)
        all_targetsadv = torch.cat([targets_x, targets_x, targets_ori], dim=0)

        input_aadv, input_badv = all_inputsadv, all_inputsadv[idx]
        target_aadv, target_badv = all_targetsadv, all_targetsadv[idx]

        mixed_inputadv = mixmatch_l * input_aadv + (1 - mixmatch_l) * input_badv
        mixed_targetadv = mixmatch_l * target_aadv + (1 - mixmatch_l) * target_badv

        logits_adv, feat_adv = model(mixed_inputadv[batch_size * 2:], adv=True, return_features=True)
        prob_adv = torch.softmax(logits_adv, dim=-1)
        y_adv = torch.log(torch.gather(prob_adv, 1, targets_uadv.view(-1, 1)).squeeze(dim=1))
        Ladv = torch.mean((prob_adv - mixed_targetadv[batch_size * 2:]) ** 2)
        pip = (normalize_flatten_features(feat_adv) - \
               normalize_flatten_features(feat_ori).detach()).norm(dim=1)
        _, targets_adv = torch.max(logits_adv[:batch_size], 1)
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = mixmatch_l * input_a + (1 - mixmatch_l) * input_b
        mixed_target = mixmatch_l * target_a + (1 - mixmatch_l) * target_b

        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx_mean = -torch.mean(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], 0)
        Lx = torch.sum(Lx_mean * class_weights)

        probs_u = torch.softmax(logits_u, dim=1)
        Lu = torch.mean((probs_u - mixed_target[batch_size * 2:])**2)
        loss = Lx + linear_rampup(epoch + batch_idx / num_iter, args.T1) * ( Lu+Ladv)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_lx.update(Lx.item(), batch_size * 2)
        losses_lu.update(Lu.item(), len(logits) - batch_size * 2)
        losses.update(loss.item(), len(logits))
        prec_adv = (targets_ux == targets_adv).float().mean()
        wandb.log({'loss/l_adv': Ladv.data.item(),
                   'loss/l_u': Lu.data.item(),
                   'loss/l_x': Lx.data.item(),
                   'prec_adv': prec_adv.item(),
                   'y_adv': y_adv.mean().cpu().detach().numpy(),
                   'y_w': y_w.mean().cpu().detach().numpy(),
                   'pip': pip.mean().cpu().detach().numpy(),
                   'l2_norm': torch.mean((inputs_adv - inputs_ori).reshape(inputs_ori.shape[0], -1).norm(dim=1)),
                   'l2_norm_his': wandb.Histogram((inputs_adv - inputs_ori).reshape(inputs_ori.shape[0], -1).norm(
                       dim=1).cpu().detach().numpy(), num_bins=512)})
    print(losses, losses_lx, losses_lu)


def splite_confident(outs, clean_targets, noisy_targets):
    probs, preds = torch.max(outs.data, 1)

    confident_correct_num = 0
    unconfident_correct_num = 0
    confident_indexs = []
    unconfident_indexs = []

    for i in range(0, len(noisy_targets)):
        if preds[i] == noisy_targets[i]:
            confident_indexs.append(i)
            if clean_targets[i] == preds[i]:
                confident_correct_num += 1
        else:
            unconfident_indexs.append(i)
            if clean_targets[i] == preds[i]:
                unconfident_correct_num += 1

    # print(getTime(), "confident and unconfident num:", len(confident_indexs), round(confident_correct_num / len(confident_indexs) * 100, 2), len(unconfident_indexs), round(unconfident_correct_num / len(unconfident_indexs) * 100, 2))
    return confident_indexs, unconfident_indexs


def update_trainloader(model, train_data, clean_targets, noisy_targets):
    predict_dataset = Semi_Unlabeled_Dataset(train_data, clean_targets, transform_train)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model)

    confident_indexs, unconfident_indexs = splite_confident(soft_outs, clean_targets, noisy_targets)
    confident_dataset = Semi_Labeled_Dataset(train_data[confident_indexs], noisy_targets[confident_indexs], transform_train)
    unconfident_dataset = Semi_Unlabeled_Dataset(train_data[unconfident_indexs], clean_targets[unconfident_indexs], transform_train)

    uncon_batch = int(args.batch_size / 2) if len(unconfident_indexs) > len(confident_indexs) else int(len(unconfident_indexs) / (len(confident_indexs) + len(unconfident_indexs)) * args.batch_size)
    con_batch = args.batch_size - uncon_batch

    labeled_trainloader = DataLoader(dataset=confident_dataset, batch_size=con_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    unlabeled_trainloader = DataLoader(dataset=unconfident_dataset, batch_size=uncon_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Loss function
    train_nums = np.zeros(args.num_class, dtype=int)
    for item in noisy_targets[confident_indexs]:
        train_nums[item] += 1

    # zeros are not calculated by mean
    # avoid too large numbers that may result in out of range of loss.
    with np.errstate(divide='ignore'):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0
        cw[cw > 3] = 3
    class_weights = torch.FloatTensor(cw).cuda()
    # print("Category", train_nums, "precent", class_weights)
    return labeled_trainloader, unlabeled_trainloader, class_weights


def noisy_refine(model, train_loader, num_layer, refine_times):
    if refine_times <= 0:
        return model
    # frezon all layers and add a new final layer
    for param in model.parameters():
        param.requires_grad = False

    model.renew_layers(num_layer)
    model.cuda()

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=args.PES_lr)
    for epoch in range(refine_times):
        train(model, train_loader, optimizer_adam, ceriation, epoch)
        _, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")

    for param in model.parameters():
        param.requires_grad = True

    return model


if args.dataset == 'cifar10' or args.dataset == 'CIFAR10':
    if args.T1 == 0:
        args.T1 = 20
    args.num_class = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10(root=args.data_path, train=True, download=True)
    test_set = CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True)
elif args.dataset == 'cifar100' or args.dataset == 'CIFAR100':
    if args.T1 == 0:
        args.T1 = 35
    args.num_class = 100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
    train_set = CIFAR100(root=args.data_path, train=True, download=True)
    test_set = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)

if args.noise_type == "symmetric":
    noise_include = True
else:
    noise_include = False
name = args.dataset + '_noise_{}_lambda_{}_bound_{}'.format(args.noise_rate, args.lambda_u, args.bound)
args.model_dir = 'results/'+name
if not os.path.exists(args.model_dir):
    os.system('mkdir -p %s' % (args.model_dir))
wandb.init(name=name, config=args, mode='offline')
ceriation = nn.CrossEntropyLoss().cuda()
data, _, noisy_labels, _, clean_labels, _ = dataset_split(train_set.data, np.array(train_set.targets), args.noise_rate, args.noise_type, args.data_percent, args.seed, args.num_class, noise_include)
train_dataset = Train_Dataset(data, noisy_labels, transform_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True)

model = create_model(num_classes=args.num_class)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
if args.optim == 'cos':
    scheduler = CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)
else:
    scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

best_test_acc = 0
for epoch in range(args.num_epochs):
    if epoch < args.T1:
        train(model, train_loader, optimizer, ceriation, epoch)
    else:
        if epoch == args.T1:
            model = noisy_refine(model, train_loader, 0, args.T2)

        labeled_trainloader, unlabeled_trainloader, class_weights = update_trainloader(model, data, clean_labels, noisy_labels)
        MixMatch_train(epoch, model, optimizer, labeled_trainloader, unlabeled_trainloader, class_weights)

    _, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")
    wandb.log({'test/acc': test_acc},commit=False)
    best_test_acc = test_acc if best_test_acc < test_acc else best_test_acc
    scheduler.step()
torch.save(model.state_dict(),
           os.path.join(args.model_dir, 'model.pth'))
print(getTime(), "Best Test Acc:", best_test_acc)
