# RoCL-Adversarial self-supervised contrastive learning
This repository is the official PyTorch implementation of "[Adversarial self supervised contrastive learning](https://arxiv.org/abs/2006.07589)" by [Minseon Kim](https://kim-minseon.github.io), [Jihoon Tack](https://jihoon-tack.github.io/) and [Sung Ju Hwang](http://www.sungjuhwang.com).

## Requirements

Currently, requires following packages
- python 3.6+
- torch 1.1+
- torchvision 0.3+
- CUDA 10.1+
- [torchlars](https://github.com/kakaobrain/torchlars) == 0.1.2 
- [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr) packages 
- [diffdist](https://github.com/ag14774/diffdist) == 0.1

## Training

To train the model(s) in the paper, run this command:

mkdir Data folder inside the RoCL
```makefolder
mkdir ./Data
```

```train
python -m torch.distributed.launch --nproc_per_node=2 rocl_train.py --ngpu 2 --batch-size=256 --model='ResNet18' --k=7 --loss_type='sim' --advtrain_type='Rep' --attack_type='linf' --name=<name-of-the-file> --regularize_to='other' --attack_to='other' --train_type='contrastive' --dataset='cifar-10'
```

## Evaluation

To evaluate my model linear evaluation and robustness, run:

```eval
./total_process.sh test <checkpoint-load> <name> <model type='ResNet18' or 'ResNet50'> <learning rate=0.1> <dataset='cifar-10' or 'cifar-100'>
```

## Results

Our model achieves the following performance on :

### Classification and robustness on CIFAR 10

| Model name         |    Accuracy     |   robustness   |
| ------------------ |---------------- | -------------- |
| RoCL ResNet18      |    83.71 %      |     40.27%     |


## Citation
```
@inproceedings{kim2020adversarial,
  title={Adversarial Self-Supervised Contrastive Learning},
  author={Minseon Kim and Jihoon Tack and Sung Ju Hwang},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2020}
}
```


