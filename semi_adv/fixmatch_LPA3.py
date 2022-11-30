import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model
from tqdm import tqdm
import torch.distributed as dist
from dataset.cifar_index import DATASET_GETTERS, mu_cifar100, std_cifar100, mu_cifar10, std_cifar10, mu_stl10, std_stl10, clamp
from utils import AverageMeter, accuracy, setup_logger
import random
import models
import pdb
import math
import torchvision
from torch.autograd import Variable
from torch import nn
from typing import List, Optional, Tuple, Union, cast
import copy
logger = logging.getLogger(__name__)
best_acc = 0


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def reconst_images(x_adv, strong_x, run):
    grid_X = torchvision.utils.make_grid(strong_x[:10].data, nrow=10, padding=2, normalize=True)
    grid_AdvX = torchvision.utils.make_grid(x_adv[:10].data, nrow=10, padding=2, normalize=True)
    grid_Delta = torchvision.utils.make_grid(x_adv[:10]-strong_x[:10].data, nrow=10, padding=2, normalize=True)
    grid = torch.cat((grid_X, grid_AdvX, grid_Delta), dim=1)
    run.log({"Batch.jpg": [
        wandb.Image(grid)]}, commit=False)
    run.log({'His/l2_norm': wandb.Histogram((x_adv - strong_x).reshape(strong_x.shape[0], -1).norm(dim=1).cpu().detach().numpy(), num_bins=512),
               }, commit=False)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_attack(model, inputs, targets_u, y_ori, flat_feat_ori, args):
    if args.dataset == 'cifar10':
        upper_limit = ((1 - mu_cifar10) / std_cifar10).to(args.device)
        lower_limit = ((0 - mu_cifar10) / std_cifar10).to(args.device)
    elif args.dataset == 'cifar100':
        upper_limit = ((1 - mu_cifar100) / std_cifar100).to(args.device)
        lower_limit = ((0 - mu_cifar100) / std_cifar100).to(args.device)
    elif args.dataset == 'stl10':
        upper_limit = ((1 - mu_stl10) / std_stl10).to(args.device)
        lower_limit = ((0 - mu_stl10) / std_stl10).to(args.device)

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

        logits_adv, feat_adv = model(inputs_adv, adv=True, return_feature=True)
        prob_adv = torch.softmax(logits_adv / args.T, dim=-1)
        y_adv = torch.log(torch.gather(prob_adv, 1, targets_u.view(-1, 1)).squeeze(dim=1))

        pip = (normalize_flatten_features(feat_adv) - \
        flat_feat_ori).norm(dim=1).mean()
        constraint = y_ori - y_adv
        loss = -pip + lam * F.relu(constraint - args.bound).mean()
        loss.backward()

        grad = perturbations.grad.data
        grad_normed = grad / \
                      (grad.reshape(grad.size()[0], -1).norm(dim=1)
                       [:, None, None, None] + 1e-8)
        with torch.no_grad():
            y_after = torch.log(torch.gather(torch.softmax(
                         model(inputs + perturbations - grad_normed * 0.1, adv=True)/args.T, dim=1),
                         1, targets_u.view(-1, 1)).squeeze(dim=1))
            dist_grads = torch.abs( y_adv - y_after
                         ) / 0.1
            norm = step_size / (dist_grads + 1e-4)
        perturbation_updates = -grad_normed * norm[:, None, None, None]

        perturbations.data = clamp(perturbations + perturbation_updates,
                                   lower_limit - inputs, upper_limit - inputs).detach()

    inputs_adv = (inputs + perturbations).detach()
    return inputs_adv


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='stl10', type=str,
                        choices=['cifar10', 'cifar100', 'stl10'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'wideresnetVar'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--our_threshold', default=0.9, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--warmup_adv', default=5, type=int, help='warm up epoch')
    parser.add_argument('--bound', default=0.1, type=float, help='bound for adversarial')
    parser.add_argument('--portion', default=0.1, type=float, help='bound for adversarial')
    parser.add_argument('--num_iterations', default=5, type=int, help='eps for adversarial')
    parser.add_argument('--lam', default=0.1, type=float, help='bound for adversarial')
    parser.add_argument("--teacher_path", type=str,
                        default='/gdata2/yangkw/semi_adv/results/cifar100_supervise/0290.ckpt.pth')
    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as nets
            model = nets.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes,
                                            bn_adv_flag=True,
                                            bn_adv_momentum=0.01)
            teacher = nets.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'wideresnetVar':
            import models.wideresnet as nets
            model = nets.build_wideresnetVar(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes,
                                            bn_adv_flag=True,
                                            bn_adv_momentum=0.01)
            teacher = nets.build_wideresnetVar(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        print("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model, teacher

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    print(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
    )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        run = wandb.init(
            config=args, name=args.out, save_code=True,
        )
        setup_logger(args.out)
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
    elif args.dataset == 'stl10':
        args.num_classes = 10
        if args.arch == 'wideresnetVar':
            args.model_depth = 28
            args.model_width = 2

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, '../data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model, _ = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    # teacher.to(args.device)
    # teacher.load_state_dict(torch.load(args.teacher_path))

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    mem_logits = Variable(torch.ones([len(unlabeled_dataset), args.num_classes], dtype=torch.int64, requires_grad=False).to(args.device) + 0.01)
    mem_tc = Variable(torch.zeros(len(unlabeled_dataset), requires_grad=False).to(args.device))
    threshold = 1
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
             model, optimizer, opt_level=args.opt_level)
    if args.local_rank != -1:
        model = DDP(model, delay_allreduce=True)

    print("***** Running training *****")
    print(f"  Task = {args.dataset}@{args.num_labeled}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Batch size per GPU = {args.batch_size}")
    print(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    print(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    if args.amp:
        from apex import amp
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x_w, targets_x, _ = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x_w, targets_x, _ = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), targets_ux, index = unlabeled_iter.next()
            except:
                _, indices = torch.sort(mem_tc, descending=True)
                kt = args.portion*len(unlabeled_dataset)
                mem_tc_copy = copy.deepcopy(mem_tc)
                threshold = mem_tc_copy[indices[int(kt)]]
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), targets_ux, index = unlabeled_iter.next()

            data_time.update(time.time() - end)
            model.train()
            all_x = torch.cat((inputs_x_w, inputs_u_w, inputs_u_s)).to(args.device)
            targets_x = targets_x.to(args.device)
            targets_ux = targets_ux.to(args.device)
            inputs_u_w = inputs_u_w.to(args.device)
            index = index.to(args.device)
            batch_size = inputs_x_w.size(0)

            with torch.no_grad():
                logits_ori, feat_ori = model(inputs_u_w, adv=True, return_feature=True)
                _, targets_uadv = torch.max(logits_ori, 1)
                flat_feat_ori = normalize_flatten_features(feat_ori)
                prob = torch.softmax(logits_ori / args.T, dim=-1)
                y_w = torch.log(torch.gather(prob, 1, targets_uadv.view(-1, 1)).squeeze(dim=1))
                at = F.kl_div(mem_logits[index].log(), prob, reduction='none').mean(dim=1)
                mask_smooth = (mem_tc[index]).lt(threshold)

            if args.world_size > 1:
                mask_smooth_all = torch.cat(GatherLayer.apply(mask_smooth), dim=0)
                run_adv = all(_.sum() > 0 for _ in mask_smooth_all.chunk(args.world_size))
                train_adv = run_adv and epoch > args.warmup_adv
            else:
                run_adv = mask_smooth.sum() > 0
                train_adv = run_adv and epoch > args.warmup_adv

            if run_adv:
                inputs_adv = get_attack(model, inputs_u_w[mask_smooth], targets_uadv[mask_smooth], \
                                        y_w[mask_smooth], flat_feat_ori[mask_smooth], args)
                optimizer.zero_grad()

            logits, _ = model(all_x, return_feature=True)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            l_ce = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = (max_probs.ge(args.threshold)).float()
            mask_1 = max_probs[mask_smooth].ge(args.our_threshold)

            l_cs = (F.cross_entropy(logits_u_s, targets_u, reduction='none')* mask).mean()
            ##
            if run_adv:
                logits_adv, feat_adv = model(inputs_adv, adv=True, return_feature=True)
                _, targets_adv = torch.max(logits_adv, 1)
                prob_adv = torch.softmax(logits_adv / args.T, dim=-1)
                y_adv = torch.log(torch.gather(prob_adv, 1, targets_u[mask_smooth].view(-1, 1)).squeeze(dim=1))
                l_adv = (F.cross_entropy(logits_adv, targets_u[mask_smooth], reduction='none')* mask_1).mean()

            if train_adv:
                loss = l_ce + l_cs + l_adv
            else:
                loss = l_ce + l_cs

            with torch.no_grad():

                prec, _ = accuracy(logits_x.data, targets_x.data, topk=(1, 5))
                prec_unlab, _ = accuracy(logits_u_w.data, targets_ux.data, topk=(1, 5))
                prec_unlab_strong, _ = accuracy(logits_u_s.data, targets_ux.data, topk=(1, 5))

                pesudo_accuracy = (targets_u == targets_ux).float()
                prec_pesudo_label = (targets_u == targets_ux).float()[max_probs.ge(args.threshold)].mean()
                if batch_idx % 10 == 0:
                    if args.local_rank in [-1, 0]:
                        hismem_tc = torch.where(torch.isnan(mem_tc), torch.full_like(mem_tc, 0), mem_tc)
                        run.log({'loss/l_cs': l_cs.data.item(),
                                 'loss/l_ce': l_ce.data.item(),
                                 'ACC/acc': prec.item(),
                                 'ACC/acc_unlab': prec_unlab.item(),
                                 'ACC/acc_unlab_strongaug': prec_unlab_strong.item(),
                                 'pesudo/prec_label': prec_pesudo_label.item(),
                                 'mask': mask.mean().item(),
                                 'Adv/mem_tc': mem_tc.mean().item(),
                                 'His/mem_tc': wandb.Histogram(hismem_tc.cpu().detach().numpy(), num_bins=512),
                                 'lr': optimizer.param_groups[0]['lr']})
                        if run_adv and mask_1.sum()>0:
                            pip = (normalize_flatten_features(feat_adv) - \
                                   normalize_flatten_features(feat_ori)[mask_smooth].detach()).norm(dim=1)
                            prec_pesudo_adv = (targets_u[mask_smooth] == targets_adv)[mask_1].float().mean()
                            l2_norm = (inputs_adv[mask_1] - (inputs_u_w[mask_smooth])[mask_1]).reshape(
                                (inputs_u_w[mask_smooth])[mask_1].shape[0], -1).norm(dim=1)
                            run.log({'loss/l_adv': l_adv.data.item(),
                                 'group1/y_adv': y_adv[mask_1].mean().cpu().detach().numpy(),
                                 'group1/y_w': (y_w[mask_smooth])[mask_1].mean().cpu().detach().numpy(),
                                 'group1/pip': pip[mask_1].mean().cpu().detach().numpy(),
                                 'group1/pesudo_acc': (pesudo_accuracy[mask_smooth])[mask_1].mean().cpu().detach().numpy(),
                                 'pesudo/prec_adv': prec_pesudo_adv.item(),
                                 'group1/num': mask_1.sum().cpu().detach().numpy(),
                                 'group1/l2_norm': torch.mean(l2_norm).cpu().detach().numpy(),
                                 'group1/l2_norm_his': wandb.Histogram(l2_norm.cpu().detach().numpy(), num_bins=512)}, commit=False)
            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            with torch.no_grad():
                if args.world_size > 1:
                    index_all = torch.cat(GatherLayer.apply(index), dim=0)
                    prob_all = torch.cat(GatherLayer.apply(prob), dim=0)
                    at_all = torch.cat(GatherLayer.apply(at), dim=0)
                else:
                    index_all = index
                    prob_all = prob
                    at_all = at
                mem_tc[index_all] = 0.01 * mem_tc[index_all] - 0.99 * at_all
                mem_logits[index_all] = prob_all

            losses_x.update(l_ce.item())
            losses_u.update(l_cs.item())
            mask_probs.update(mask.mean().item())
            scheduler.step()

            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s.  Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}.  Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)
            run.log({'test/1.test_acc': test_acc,
                         'test/2.test_loss': test_loss})
            #reconst_images(inputs_adv, inputs_u_w, run, mask_1, mask_2, mask_3)
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'mem_logits': mem_logits,
                'mem_tc': mem_tc,
            }, is_best, args.out)

            test_accs.append(test_acc)
            print('Best top-1 acc: {:.2f}'.format(best_acc))
            print('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        run.finish()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    print(epoch)
    print("top-1 acc: {:.2f}".format(top1.avg))
    print("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
