#!/usr/bin/env python3 -u

from __future__ import print_function

import csv
import os

import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim
import data_loader
import model_loader

from attack_lib import FastGradientSignUntargeted,RepresentationAdv

from models.projector import Projector
from argument import parser, print_args
from utils import progress_bar, checkpoint, AverageMeter, accuracy

from loss import pairwise_similarity, NT_xent
from torchlars import LARS
from warmup_scheduler import GradualWarmupScheduler

args = parser()

def print_status(string):
    if args.local_rank % ngpus_per_node == 0:
        print(string)

ngpus_per_node = torch.cuda.device_count()
if args.ngpu>1:
    multi_gpu=True
elif args.ngpu==1:
    multi_gpu=False
else:
    assert("Need GPU....")
if args.local_rank % ngpus_per_node == 0:
    print_args(args)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

world_size = args.ngpu
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
)

# Data
print_status('==> Preparing data..')
if not (args.train_type=='contrastive'):
    assert('wrong train phase...')
else:
    trainloader, traindst, testloader, testdst ,train_sampler = data_loader.get_dataset(args)

# Model
print_status('==> Building model..')
torch.cuda.set_device(args.local_rank)
model = model_loader.get_model(args)

if args.model=='ResNet18':
    expansion=1
elif args.model=='ResNet50':
    expansion=4
else:
    assert('wrong model type')
projector = Projector(expansion=expansion)

if 'Rep' in args.advtrain_type:
    Rep_info = 'Rep_attack_ep_'+str(args.epsilon)+'_alpha_'+ str(args.alpha) + '_min_val_' + str(args.min) + '_max_val_' + str(args.max) + '_max_iters_' + str(args.k) + '_type_' + str(args.attack_type) + '_randomstart_' + str(args.random_start)
    args.name += Rep_info
    
    print_status("Representation attack info ...")
    print_status(Rep_info)
    Rep = RepresentationAdv(model, projector, epsilon=args.epsilon, alpha=args.alpha, min_val=args.min, max_val=args.max, max_iters=args.k, _type=args.attack_type, loss_type=args.loss_type, regularize = args.regularize_to)
else:
    assert('wrong adversarial train type')

# Model upload to GPU # 
model.cuda()
projector.cuda()

model       = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model       = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)
projector   = torch.nn.parallel.DistributedDataParallel(
                projector,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)

cudnn.benchmark = True
print_status('Using CUDA..')

# Aggregating model parameter & projection parameter #
model_params = []
model_params += model.parameters()
model_params += projector.parameters()

# LARS optimizer from KAKAO-BRAIN github "pip install torchlars" or git from https://github.com/kakaobrain/torchlars
base_optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer       = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

# Cosine learning rate annealing (SGDR) & Learning rate warmup git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.lr_multiplier, total_epoch=10, after_scheduler=scheduler_cosine)

def train(epoch):
    if args.local_rank % ngpus_per_node == 0:
        print('\nEpoch: %d' % epoch)

    model.train()
    projector.train()

    train_sampler.set_epoch(epoch)
    scheduler_warmup.step()

    total_loss = 0
    reg_simloss = 0
    reg_loss = 0

    for batch_idx, (ori, inputs_1, inputs_2, label) in enumerate(trainloader):
        ori, inputs_1, inputs_2 = ori.cuda(), inputs_1.cuda() ,inputs_2.cuda()

        if args.attack_to=='original':
            attack_target = inputs_1
        else:
            attack_target = inputs_2

        if 'Rep' in args.advtrain_type :
            advinputs, adv_loss = Rep.get_loss(original_images=inputs_1, target = attack_target, optimizer=optimizer, weight= args.lamda, random_start=args.random_start)
            reg_loss    += adv_loss.data

        if not (args.advtrain_type == 'None'):
            inputs = torch.cat((inputs_1, inputs_2, advinputs))
        else:
            inputs = torch.cat((inputs_1, inputs_2))
        
        outputs = projector(model(inputs))
        similarity, gathered_outputs = pairwise_similarity(outputs, temperature=args.temperature, multi_gpu=multi_gpu, adv_type = args.advtrain_type) 
        
        simloss  = NT_xent(similarity, args.advtrain_type)
        
        if not (args.advtrain_type=='None'):
            loss = simloss + adv_loss
        else:
            loss = simloss
        
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.data
        reg_simloss += simloss.data
        
        optimizer.step()

        if (args.local_rank % ngpus_per_node == 0):
            if 'Rep' in args.advtrain_type:
                progress_bar(batch_idx, len(trainloader),
                             'Loss: %.3f | SimLoss: %.3f | Adv: %.2f'
                             % (total_loss / (batch_idx + 1), reg_simloss / (batch_idx + 1), reg_loss / (batch_idx + 1)))
            else:
                progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Adv: %.3f'
                         % (total_loss/(batch_idx+1), reg_simloss/(batch_idx+1)))
        
    return (total_loss/batch_idx, reg_simloss/batch_idx)


def test(epoch, train_loss):
    model.eval()
    projector.eval()

    # Save at the last epoch #       
    if epoch == args.epoch - 1 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model, train_loss, epoch, args, optimizer)
        checkpoint(projector, train_loss, epoch, args, optimizer, save_name_add='_projector')
       
    # Save at every 100 epoch #
    elif epoch % 100 == 0 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model, train_loss, epoch, args, optimizer, save_name_add='_epoch_'+str(epoch))
        checkpoint(projector, train_loss, epoch, args, optimizer, save_name_add=('_projector_epoch_' + str(epoch)))


# Log and saving checkpoint information #
if not os.path.isdir('results') and args.local_rank % ngpus_per_node == 0:
    os.mkdir('results')

args.name += (args.train_type + '_' +args.model + '_' + args.dataset + '_b' + str(args.batch_size)+'_nGPU'+str(args.ngpu)+'_l'+str(args.lamda))
loginfo = 'results/log_' + args.name + '_' + str(args.seed)
logname = (loginfo+ '.csv')
print_status('Training info...')
print_status(loginfo)

##### Log file #####
if args.local_rank % ngpus_per_node == 0:
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss'])

print(args.name)

##### Training #####
for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss = train(epoch)
    test(epoch, train_loss)

    if args.local_rank % ngpus_per_node == 0:
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss.item(), reg_loss.item()])


