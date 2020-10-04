#!/usr/bin/env python3 -u

from __future__ import print_function

import argparse
import csv
import os
import json
import copy

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import data_loader
import model_loader
import models
from models.projector import Projector

from argument import linear_parser, print_args
from utils import progress_bar, checkpoint
from collections import OrderedDict
from attack_lib import FastGradientSignUntargeted
from loss import pairwise_similarity, NT_xent

args = linear_parser()
use_cuda = torch.cuda.is_available()
if use_cuda:
    ngpus_per_node = torch.cuda.device_count()

if args.local_rank % ngpus_per_node==0:
    print_args(args)

def print_status(string):
    if args.local_rank % ngpus_per_node == 0:
        print(string)

print_status(torch.cuda.device_count())
print_status('Using CUDA..')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print_status('==> Preparing data..')
if not (args.train_type=='linear_eval'):
    assert('wrong train phase...')
else:
    trainloader, traindst, testloader, testdst  = data_loader.get_dataset(args)

if args.dataset == 'cifar-10' or args.dataset=='mnist':
    num_outputs = 10
elif args.dataset == 'cifar-100':
    num_outputs = 100

if args.model == 'ResNet50':
    expansion = 4
else:
    expansion = 1

# Model
print_status('==> Building model..')
train_type  = args.train_type

def load(args, epoch):
    model = model_loader.get_model(args)

    if epoch == 0:
        add = ''
    else:
        add = '_epoch_'+str(epoch)

    checkpoint_ = torch.load(args.load_checkpoint+add)

    new_state_dict = OrderedDict()
    for k, v in checkpoint_['model'].items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    
    if args.ss:
        projector = Projector(expansion=expansion)
        checkpoint_p = torch.load(args.load_checkpoint+'_projector'+add)
        new_state_dict = OrderedDict()
        for k, v in checkpoint_p['model'].items():
            name = k[7:]
            new_state_dict[name] = v
        projector.load_state_dict(new_state_dict)
    
    if args.dataset=='cifar-10':
        Linear = nn.Sequential(nn.Linear(512*expansion, 10))
    elif args.dataset=='cifar-100':
        Linear = nn.Sequential(nn.Linear(512*expansion, 100))

    model_params = []
    if args.finetune:
        model_params += model.parameters()
        if args.ss:
            model_params += projector.parameters()
    model_params += Linear.parameters()
    loptim = torch.optim.SGD(model_params, lr = args.lr, momentum=0.9, weight_decay=5e-4)
   
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        ngpus_per_node = torch.cuda.device_count()
        model.cuda()
        Linear.cuda()
        model = nn.DataParallel(model)
        Linear = nn.DataParallel(Linear)
        if args.ss:
            projector.cuda()
            projector = nn.DataParallel(projector)
    else:
        assert("Need to use GPU...")

    print_status('Using CUDA..')
    cudnn.benchmark = True

    if args.adv_img:
        attack_info = 'Adv_train_epsilon_'+str(args.epsilon)+'_alpha_'+ str(args.alpha) + '_min_val_' + str(args.min) + '_max_val_' + str(args.max) + '_max_iters_' + str(args.k) + '_type_' + str(args.attack_type) + '_randomstart_' + str(args.random_start)
        print_status("Adversarial training info...")
        print_status(attack_info)

        attacker = FastGradientSignUntargeted(model, linear=Linear, epsilon=args.epsilon, alpha=args.alpha, min_val=args.min, max_val=args.max, max_iters=args.k, _type=args.attack_type)

    if args.adv_img:
        if args.ss:
            return model, Linear, projector, loptim, attacker
        return model, Linear, 'None', loptim, attacker
    if args.ss:
        return model, Linear, projector, loptim, 'None'
    return model, Linear, 'None', loptim, 'None'

criterion = nn.CrossEntropyLoss()


def linear_train(epoch, model, Linear, projector, loptim, attacker=None):
    Linear.train()
    if args.finetune:
        model.train()
        if args.ss:
            projector.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (ori, inputs, inputs_2, target) in enumerate(trainloader):
        ori, inputs_1, inputs_2, target = ori.cuda(), inputs.cuda(), inputs_2.cuda(), target.cuda()
        input_flag = False
        if args.trans:
            inputs = inputs_1
        else:
            inputs = ori

        if args.adv_img:
            advinputs      = attacker.perturb(original_images=inputs, labels=target, random_start=args.random_start)
        
        if args.clean:
            total_inputs = inputs
            total_targets = target
            input_flag = True
        
        if args.ss:
            total_inputs = torch.cat((inputs, inputs_2))
            total_targets = torch.cat((target, target))

        if args.adv_img:
            if input_flag:
                total_inputs = torch.cat((total_inputs, advinputs))
                total_targets = torch.cat((total_targets, target))
            else:
                total_inputs = advinputs
                total_targets = target
                input_flag = True

        if not input_flag:
            assert('choose the linear evaluation data type (clean, adv_img)')

        feat   = model(total_inputs)
        if args.ss:
            output_p = projector(feat)
            B = ori.size(0)

            similarity, _ = pairwise_similarity(output_p[:2*B,:2*B], temperature=args.temperature, multi_gpu=False, adv_type = 'None')
            simloss  = NT_xent(similarity, 'None')
        
        output = Linear(feat)

        _, predx = torch.max(output.data, 1)
        loss = criterion(output, total_targets)
        
        if args.ss:
            loss += simloss

        correct += predx.eq(total_targets.data).cpu().sum().item()
        total += total_targets.size(0)
        acc = 100.*correct/total

        total_loss += loss.data

        loptim.zero_grad()
        loss.backward()
        loptim.step()
        
        progress_bar(batch_idx, len(trainloader),
                    'Loss: {:.4f} | Acc: {:.2f}'.format(total_loss/(batch_idx+1), acc))

    print ("Epoch: {}, train accuracy: {}".format(epoch, acc))

    return acc, model, Linear, projector, loptim

def test(model, Linear):
    global best_acc

    model.eval()
    Linear.eval()

    test_loss = 0
    correct = 0
    total = 0

    for idx, (image, label) in enumerate(testloader):
        img = image.cuda()
        y = label.cuda()

        out = Linear(model(img))

        _, predx = torch.max(out.data, 1)
        loss = criterion(out, y)

        correct += predx.eq(y.data).cpu().sum().item()
        total += y.size(0)
        acc = 100.*correct/total

        test_loss += loss.data
        if args.local_rank % ngpus_per_node == 0:
            progress_bar(idx, len(testloader),'Testing Loss {:.3f}, acc {:.3f}'.format(test_loss/(idx+1), acc))

    print ("Test accuracy: {0}".format(acc))

    return (acc, model, Linear)

def adjust_lr(epoch, optim):
    lr = args.lr
    if args.dataset=='cifar-10' or args.dataset=='cifar-100':
        lr_list = [30,50,100]
    if epoch>=lr_list[0]:
        lr = lr/10
    if epoch>=lr_list[1]:
        lr = lr/10
    if epoch>=lr_list[2]:
        lr = lr/10
    
    for param_group in optim.param_groups:
        param_group['lr'] = lr

##### Log file for training selected tasks #####
if not os.path.isdir('results'):
    os.mkdir('results')

args.name += ('_Evaluate_'+ args.train_type + '_' +args.model + '_' + args.dataset)
loginfo = 'results/log_generalization_' + args.name + '_' + str(args.seed)
logname = (loginfo+ '.csv')

with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train acc','test acc'])

if args.epochwise:
    for k in range(100,1000,100):
        model, linear, projector, loptim, attacker = load(args, k)
        print('loading.......epoch ', str(k))
        ##### Linear evaluation #####
        for i in range(args.epoch):
            print('Epoch ', i)
            train_acc, model, linear, projector, loptim = linear_train(i, model, linear, projector, loptim, attacker)
            test_acc, model, linear = test(model, linear)
            adjust_lr(i, loptim)

        checkpoint(model, test_acc, args.epoch, args, loptim, save_name_add='epochwise'+str(k))
        checkpoint(linear, test_acc, args.epoch, args, loptim, save_name_add='epochwise'+str(k)+'_linear')
        if args.local_rank % ngpus_per_node == 0:
            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([k, train_acc, test_acc])

model, linear, projector, loptim, attacker = load(args, 0)

##### Linear evaluation #####
for epoch in range(args.epoch):
    print('Epoch ', epoch)

    train_acc, model, linear, projector, loptim = linear_train(epoch, model=model, Linear=linear, projector=projector, loptim=loptim, attacker=attacker)
    test_acc, model, linear = test(model, linear)
    adjust_lr(epoch, loptim)

    if args.local_rank % ngpus_per_node == 0:
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_acc, test_acc])

checkpoint(model, test_acc, args.epoch, args, loptim)
checkpoint(linear, test_acc, args.epoch, args, loptim, save_name_add='_linear')

if args.local_rank % ngpus_per_node == 0:
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([1000, train_acc, test_acc])
