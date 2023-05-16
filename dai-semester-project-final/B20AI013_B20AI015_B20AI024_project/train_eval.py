import sys
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch import optim
from torch.nn import DataParallel
from tqdm import tqdm

from train_cel import N, enable_ln_loss, enable_ce_loss, LN_scaling


##### refactored and edited this stuff
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
        
    with torch.no_grad():
        for i, (orig, adv, label) in enumerate(tqdm(data_loader)):
            adv = Variable(adv.cuda(non_blocking = True), volatile = True)
            orig = Variable(orig.cuda(non_blocking = True), volatile = True)
            
            label = Variable(label.cuda(non_blocking = True))
            outputs = net(adv, defense = True)
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
    
    
class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass