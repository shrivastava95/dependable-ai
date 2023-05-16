import torch
import numpy as np
import os
import sys
import argparse
from importlib import import_module
# from train_eval import Logger #!
import shutil
import time
from matplotlib import pyplot as plt

from train_cel import name_text

from torch import optim
from data_new import DefenseDataset
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
import collections

from torch.autograd import Variable

from train_cel import N, enable_ln_loss, enable_ce_loss, LN_scaling
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch defense model')
parser.add_argument('--exp', '-e', metavar='MODEL', default='sample',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--print-iter', default=0, type=int, metavar='I',
                    help='print per iter')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--debug', action = 'store_true',
                    help='debug mode')
parser.add_argument('--test', default='0', type=int, metavar='T',
                    help='test mode')
parser.add_argument('--test_e4', default=0, type=int, metavar = 'T', 
                    help= 'test eps 4')
parser.add_argument('--defense', default=1, type=int, metavar='T',
                    help='test mode')
parser.add_argument('--optimizer', default='adam', type=str, metavar='O',
                    help='optimizer')



def test(phase, net, loss_fn, data_loader, requires_control = True):
    # ## log outputs to noise folder. store the latest epoch only
    # noise = {
    #     'orig': [],
    #     'adv': [],
    #     'pred_noise': [],
    #     'actual_noise': [],
    #     'label': [],
    # }

# 'noise/noise_cifar10_resnet_pgd_train.pt'
    start_time = time.time()
    net.eval()


    acc = []
    loss = []
    if requires_control:
        orig_acc = []
        orig_loss = []
    
    from torchvision import transforms as tf
    with torch.no_grad():
        for i, (orig, adv, label) in enumerate(tqdm(data_loader)):
            adv = Variable(adv.cuda(non_blocking = True), volatile = True)
            orig = Variable(orig.cuda(non_blocking = True), volatile = True)
            
            label = Variable(label.cuda(non_blocking = True))
            outputs = net(adv, defense = True)
            ###########################################################
            denoised = net.denoise_fn(adv)
            
            features = adv
            print('BRUH WE IN  denoiseer', len(net.features))
            for layer in net.features[:5]:
                print(features.shape)
                features = layer(features)

            _, _, scores = net(denoised)

            for class_id in range(10):
                loss = torch.Tensor([0])[0]
                loss.requires_grad = True
                loss = loss + scores[0][class_id]
                loss.backward(retain_graph=True)


                alphas = torch.mean(features.grad, dim=(2, 3)).unsqueeze(2).unsqueeze(2)
                print(features.shape, alphas.shape)
                maps = torch.sum(tf.Resize([32, 32])(features * alphas), dim=1)
                maps = torch.nn.ReLU()(maps)
                plt.imshow(maps[0].clone().cpu().detach().numpy(), cmap='hot', interpolation='nearest')
                plt.show()


            
            

            ###########################################################
            logits = outputs[-1]

            
            acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
            if enable_ce_loss:
                l = loss_fn(logits, label)

            #### ishaan: LN loss
            if enable_ln_loss:
                ln = torch.mean(torch.mean(torch.pow((torch.abs(outputs[0] - orig)),  N)  / N, dim=(0, 1, 2, 3)))
                if enable_ce_loss:
                    l = l + LN_scaling * ln#!
                else:
                    l = LN_scaling* ln
            ####

            # loss.append(l.data[0])
            loss.append(l.item())
            l = l.detach()
            pred_orig = outputs[0]
            pred_noise = adv - pred_orig
            actual_noise = adv - orig

            
            if requires_control:
                orig = Variable(orig.cuda(non_blocking = True), volatile = True)
                label = Variable(label.data, volatile = True)
                logits = net(orig)[-1]
                orig_acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
                l = loss_fn(logits, label)
                # orig_loss.append(l.data[0])
                orig_loss.append(l.item())
                l = l.detach()
            
            # if requires_control:
            #     print(f'{phase} mid-epoch: loss: {loss[-1]}, acc: {acc[-1]}, orig_loss: {orig_loss[-1]} orig_acc: {orig_acc[-1]}')
            # else:
            #     print(f'{phase} mid-epoch: loss: {loss[-1]}, acc: {acc[-1]}')
            
            break
                

    acc = np.mean(acc)
    loss = np.mean(loss)
    if requires_control:
        orig_acc = np.mean(orig_acc)
        orig_loss = np.mean(orig_loss)
    end_time = time.time()
    dt = end_time - start_time
    
    if requires_control:
        print(f'{phase}: loss {loss:.5f}, acc {acc:.3f}, orig_loss {orig_loss:.5f}, orig_acc {orig_acc:.3f}, time {dt:3.1f}')
    else: 
        print(f'{phase}: loss {loss:.5f}, acc {acc:.3f}, time {dt:3.1f}')
    
    # print





def main():

    global args
    args = parser.parse_args()

    modelpath = os.path.join(os.path.abspath('../Exps'),args.exp)
    # train_data = np.load(os.path.join(modelpath,'train_split.npy'))
    # val_data = np.load(os.path.join(modelpath,'val_split.npy'))
    # with open(os.path.join(modelpath,'train_attack.txt'),'r') as f:
    #     train_attack = f.readlines()
    # train_attack = [attack.split(' ')[0].split(',')[0].split('\n')[0] for attack in train_attack]
    sys.path.append(modelpath)
    model = import_module('model')
    config, net = model.get_model()
    net = net.net#!
    state_dict = torch.load(f'noise/noise_cifar10_resnet_pgd_train{name_text}.pt')['net.state_dict()']
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join(modelpath,'results', save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join(modelpath,'results',  exp_id)
        else:
            save_dir = os.path.join(modelpath,'results', save_dir)
    print(save_dir)
        
    if args.debug:
        net = net.cuda()
    else:
        net = DataParallel(net).cuda()
    loss_fn = loss_fn.cuda()
    cudnn.benchmark = True
    
    dataset = DefenseDataset(config,  'train', 'pgd')
    # dataset = DefenseDataset(config,  'train', train_data, train_attack)           # ishaan: changed the old dataset import class to the new one. 
    train_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True)
    # dataset = DefenseDataset(config,  'val', train_attack)
    # # dataset = DefenseDataset(config,  'val', val_data, train_attack)           # ishaan: changed the old dataset import class to the new one. Also, remember to add the val files in the pretraining.
    # val_loader = DataLoader(
    #     dataset,
    #     batch_size = args.batch_size,
    #     shuffle = True,
    #     num_workers = args.workers,
    #     pin_memory = True)
    dataset = DefenseDataset(config, 'test', 'pgd')
    test_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True,
    )

    if isinstance(net, DataParallel):
        params = net.module.denoise.parameters()
    else:
        params = net.denoise.parameters()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            params,
            lr = args.lr,
            momentum = 0.9,
            weight_decay = args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            params,
            lr = args.lr,
            weight_decay = args.weight_decay)
    else:
        exit('Wrong optimizer')

    def get_lr(epoch):
        if epoch <= args.epochs * 0.6:
            return args.lr
        elif epoch <= args.epochs * 0.9:
            return args.lr * 0.1
        else:
            return args.lr * 0.01

    # for epoch in range(start_epoch, args.epochs + 1):
    requires_control = True#epoch == start_epoch
    print()
    test('test', net, loss_fn, test_loader, requires_control = requires_control)
    # test('train', net, loss_fn, train_loader, requires_control = requires_control)



if __name__ == '__main__':
    main()