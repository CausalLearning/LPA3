# LPA3

Official implementation:
- [Adversarial Auto-Augment with Label Preservation: A Representation Learning Principle Guided Approach](https://arxiv.org/pdf/2211.00824.pdf), NeurIPS 2022. 

In the project, we release the application of LPA3 to FixMatch and PES as an illustration, and you can apply LPA3 to your own tasks.

## Requirments
* Python 3.8
* PyTorch 1.7
* Torchvision
* Wandb
* Apex
For details, see requirements.txt.

## FixMatch
```
cd FixMatch
```
* To train baseline FixMatch on CIFAR10, CIFAR100 and STL-10:
```
python -m torch.distributed.launch --nproc_per_node 4 fixmatch.py --seed 1 --dataset cifar10 --num-labeled 40 --expand-labels --amp --opt_level O2 --out ./results/baseline_cifar10_40_s1 --batch-size 16;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch.py --seed 1 --dataset cifar10 --num-labeled 250 --expand-labels --amp --opt_level O2 --out ./results/baseline_cifar10_250_s1 --batch-size 16;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch.py --seed 1 --dataset cifar10 --num-labeled 4000 --expand-labels --amp --opt_level O2 --out ./results/baseline_cifar10_4000_s1 --batch-size 16;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch.py --seed 1 --dataset cifar100 --num-labeled 400 --expand-labels --amp --opt_level O2 --wdecay 0.001 --out ./results/baseline_cifar100_400_s1 --batch-size 16;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch.py --seed 1 --dataset cifar100 --num-labeled 2500 --expand-labels --amp --opt_level O2 --wdecay 0.001 --out ./results/baseline_cifar100_2500_s1 --batch-size 16;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch.py --seed 1 --dataset cifar100 --num-labeled 10000 --expand-labels --amp --opt_level O2 --wdecay 0.001 --out ./results/baseline_cifar100_10000_s1 --batch-size 16;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch.py --arch 'wideresnetVar' --seed 1 --dataset stl10 --expand-labels --amp --opt_level O2 --out ./results/stl10_s1_baseline --batch-size 16;
```
* To train FixMatch with LPA3 on CIFAR10, CIFAR100 and STL-10:
```
python -m torch.distributed.launch --nproc_per_node 4 fixmatch_LPA3.py  --seed 1 --dataset cifar10 --num-labeled 40 --expand-labels --amp --opt_level O2 --out ./results/cifar10_40_lpa3 --batch-size 16 --bound 0.002 --lam 1 --ratio 0.9;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch_LPA3.py  --seed 1 --dataset cifar10 --num-labeled 250 --expand-labels --amp --opt_level O2 --out ./results/cifar10_250_lpa3 --batch-size 16 --bound 0.002 --lam 1 --ratio 0.9;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch_LPA3.py  --seed 1 --dataset cifar10 --num-labeled 4000 --expand-labels --amp --opt_level O2 --out ./results/cifar10_4000_lpa3 --batch-size 16 --bound 0.002 --lam 1 --ratio 0.9;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch_LPA3.py  --seed 1 --dataset cifar100 --num-labeled 400 --expand-labels --amp --opt_level O2 --wdecay 0.001 --out ./results/cifar100_400_lpa3 --batch-size 16 --bound 0.02 --lam 1 --ratio 0.9;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch_LPA3.py  --seed 1 --dataset cifar100 --num-labeled 2500 --expand-labels --amp --opt_level O2 --wdecay 0.001 --out ./results/cifar100_2500_lpa3 --batch-size 16 --bound 0.02 --lam 1 --ratio 0.9;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch_LPA3.py  --seed 1 --dataset cifar100 --num-labeled 10000 --expand-labels --amp --opt_level O2 --wdecay 0.001 --out ./results/cifar100_10000_lpa3 --batch-size 16 --bound 0.02 --lam 1 --ratio 0.9;
python -m torch.distributed.launch --nproc_per_node 4 fixmatch_LPA3.py --arch 'wideresnetVar' --seed 1 --dataset stl10 --expand-labels --amp --opt_level O2 --out ./results/stl10_lpa3 --batch-size 16 --bound 0.002 --lam 1 --ratio 0.9;
```

## PES
```
cd PES
```
* To train PES baseline on CIFAR10 and CIFAR100:
```
python PES.py --dataset cifar10 --data_path ../data/ --lambda_u 15 --noise_rate 0.5
python PES.py --dataset cifar10 --data_path ../data/ --lambda_u 25 --noise_rate 0.8
python PES.py --dataset cifar10 --data_path ../data/ --lambda_u 25 --noise_rate 0.9
python PES.py --dataset cifar100 --data_path ../data/ --lambda_u 75 --noise_rate 0.5
python PES.py --dataset cifar100 --data_path ../data/ --lambda_u 100 --noise_rate 0.8
python PES.py --dataset cifar100 --data_path ../data/ --lambda_u 100 --noise_rate 0.9
```
* To train PES with LPA3 on CIFAR10 and CIFAR100:
```
python PES_LPA3.py --dataset cifar10 --data_path ../data/  --noise_rate 0.5 --lambda_u 7.5
python PES_LPA3.py --dataset cifar10 --data_path ../data/  --noise_rate 0.8 --lambda_u 25
python PES_LPA3.py --dataset cifar10 --data_path ../data/  --noise_rate 0.9 --lambda_u 25 --bound 0.002
python PES_LPA3.py --dataset cifar100 --data_path ../data/  --noise_rate 0.5 --lambda_u 37.5
python PES_LPA3.py --dataset cifar100 --data_path ../data/  --noise_rate 0.8 --lambda_u 100
python PES_LPA3.py --dataset cifar100 --data_path ../data/  --noise_rate 0.9 --lambda_u 100
```
