from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml

import wandb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam, lr_scheduler
from torchvision import transforms, datasets
from torchvision.models import resnet50
from torchvision.datasets import DatasetFolder, ImageFolder

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import MyImageFolder
# from networks.resnet_big import SupConResNet, LMCLResNet
# from networks.layers import MarginCosineProduct, cosine_sim
# from losses import SupConLoss
# from evaluate_lmcl import validate

from models import SupConResNet, LMCLResNet, LinearClassifier
from losses import SupConLoss, LargeMarginCosLoss

# from train import set_model, set_optimizer, set_save
from train import set_optimizer, set_save
  

def set_loader(config):
    if config['dataset'] == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif config['dataset'] == 'cifar100' or config['dataset'] == 'path':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif config.get('mean') is not None and config.get('std') is not None:
        mean = eval(config['mean'])
        std = eval(config['std'])
    else:
        raise ValueError('dataset not supported: {}'.format(config['dataset']))
        
    normalize = transforms.Normalize(mean=mean, std=std)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=config['image_size'], scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.3, 0.15, 0.1, 0.1)
        ], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(31, 2)], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(size=(config['image_size'], config['image_size'])), # support batching
        transforms.ToTensor(),
        normalize,
    ])
    
    
    if config['dataset'] == 'cifar10':
        train_dataset = datasets.CIFAR10(root=config['data_folder'],
                                         transform=train_transform,
                                         download=True)
    elif config['dataset'] == 'cifar100':
        train_dataset = datasets.CIFAR100(root=config['data_folder'],
                                          transform=train_transform,
                                          download=True)
    elif config['dataset'] == 'path':
        train_dataset = MyImageFolder(root=config['data_folder']+"/train",
                                            transform=train_transform)
#         train_dataset = ImageFolder(root=config['data_folder']+"/train",
#                                             transform=train_transform)
        val_dataset = MyImageFolder(root=config['data_folder']+"/val",
                                            transform=val_transform)
#         val_dataset = ImageFolder(root=config['data_folder']+"/val",
#                                             transform=val_transform)
    
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None),
        num_workers=config['num_workers'], pin_memory=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config['batch_size'], shuffle=False,
    num_workers=config['num_workers'], pin_memory=True)

    return train_loader, val_loader


def finetune(train_loader, val_loader, model, classifier, criterion, optimizer, scheduler):
    """one epoch classifier finetuning"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()
    val_losses = AverageMeter()
    train_top1 = AverageMeter()
    val_top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        outputs = classifier(model.encode(images))

        loss = criterion(outputs, labels)
        
        # update metric
        train_losses.update(loss.item(), bsz)

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        train_top1.update(acc1[0], bsz)
        
        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # scheduler
        if scheduler is not None:
            if config['scheduler'] != 'plateau':
                scheduler.step()
            else:
                scheduler.step(acc1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # compute loss
            outputs = classifier(model.encode(images))
            loss = criterion(outputs, labels)
            
            # update metric
            val_losses.update(loss.item(), bsz)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            val_top1.update(acc1[0], bsz)
        
    return train_losses.avg, train_top1.avg, val_losses.avg, val_top1.avg
        
        

def set_model(config):
    if config['model'] == 'lmcl':
        model = LMCLResNet(**config['model_args'])
        loss = nn.CrossEntropyLoss()
    elif config['model'] == 'supcon':
        model = SupConResNet(**config['model_args'])
        loss = torch.nn.CrossEntropyLoss()
        classifier = LinearClassifier(**config['classifier_args'])
        
    if config.get('load'):
        print('Loading pretrained model from: ', config['load'])
        model.load_state_dict(torch.load(config['load'])['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()
        classifier = classifier.cuda()
    return model, classifier, loss


def set_save(config):
    # set the path according to the environment
    if not os.path.isdir('identification/save'):
        os.makedirs('identification/save')
    save_model_path = 'identification/save/{}_models'.format(config['model'])

    model_name = 'model_{}_load_pt_encoder_{}_optimizer_{}_bs_{}'.format(config['model'], config['model_args']['load_pt_encoder'], config['optimizer'], config['batch_size'])
    if config.get('scheduler') is not None:
        model_name += '_scheduler_{}'.format(config['scheduler'])

    save_model_path = os.path.join(save_model_path, model_name)
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
        
    return save_model_path, model_name

def main(config):

    best_acc = 0
    best_epoch = 0
    
    ### Dataset and DataLoader ###
    train_loader, val_loader = set_loader(config)

    ### Model and Loss ###
#     model, criterion = set_model(config)

#     classifier = LinearClassifier()

    model, classifier, criterion = set_model(config)
    
    ### Optimizer ###
    optimizer, scheduler = set_optimizer(config, classifier.parameters())

    ### Save ###
    save_model_path, model_name = set_save(config)
    
    wandb.init(project='cv_chimpanzee', name='finetune__'+model_name, config=config)

    # training routine
    for epoch in range(1, config['epochs'] + 1):

        time1 = time.time()
        train_loss, train_acc, val_loss, val_acc = finetune(train_loader, val_loader, model, classifier, criterion, optimizer, scheduler)
        time2 = time.time()
        # eval
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
        
        if epoch % config['print_freq'] == 0:
            wandb.log({'epoch': epoch, 'train_time': time2 - time1, 
                       'learning_rate': optimizer.param_groups[0]['lr'],
                       'train_loss': train_loss, 'train_acc': train_acc,
                       'val_loss': val_loss, 'val_acc': val_acc,
            })
            print('epoch {}, train time {:.2f}, train_loss {:.2f}, train_acc {:.2f}; val_loss {:.2f}, val_acc {:.2f}'.format(epoch, time2 - time1, train_loss, train_acc, val_loss, val_acc))


        if epoch % config['save_freq'] == 0:
            save_file = os.path.join(
                save_model_path, 'finetune_ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, config, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        save_model_path, 'last.pth')
    save_model(model, optimizer, config, config['epochs'], save_file)
    
    wandb.log({'best_acc': best_acc, 'best_epoch': best_epoch})
    
    print('best validation accuracy: {:.2f}, epoch: {}'.format(best_acc, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    main(config)
