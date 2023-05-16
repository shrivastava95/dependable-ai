import sys
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch import optim
from torch.nn import DataParallel
from tqdm import tqdm

name_text = 'L2'
enable_ln_loss = True
enable_ce_loss = False
N = 2
LN_scaling = 100000

def train(epoch, net, loss_fn, data_loader, optimizer, get_lr, requires_control = True):
    ## log outputs to noise folder. store the latest epoch only
    noise = {
        'orig': [],
        'adv': [],
        'pred_noise': [],
        'actual_noise': [],
        'label': [],
        'net.state_dict()': None,
    }


    start_time = time.time()
    net.eval()
    if isinstance(net, DataParallel):
        net.module.denoise.train()
    else:
        net.denoise.train()

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    acc = []
    loss = []
    if requires_control:
        orig_acc = []
        orig_loss = []
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

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # loss.append(l.data[0])
        loss.append(l.item())
        pred_orig = outputs[0]
        pred_noise = adv - pred_orig
        actual_noise = adv - orig
        noise['adv'].append(adv.detach().cpu())
        noise['label'].append(label.detach().cpu())
        noise['orig'].append(orig.detach().cpu())
        noise['pred_noise'].append(pred_noise.detach().cpu())
        noise['actual_noise'].append(actual_noise.detach().cpu())
        
        if requires_control:
            label = Variable(label.data, volatile = True)
            logits = net(orig)[-1]
            orig_acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
            l = loss_fn(logits, label)
            # orig_loss.append(l.data[0])
            orig_loss.append(l.item())
            

    acc = np.mean(acc)
    loss = np.mean(loss)
    if requires_control:
        orig_acc = np.mean(orig_acc)
        orig_loss = np.mean(orig_loss)
    end_time = time.time()
    dt = end_time - start_time
    
    if requires_control:
        print(f'train: Epoch {epoch:<3} (lr {lr:<10.6f}): loss {loss:<7.5f}, acc {acc:<5.3f}, orig_loss {orig_loss:<7.5f}, orig_acc {orig_acc:<5.3f}, time {dt:<10.1f}')
    else: 
        print(f'train: Epoch {epoch:<3} (lr {lr:<10.6f}): loss {loss:<7.5f}, acc {acc:<5.3f}, time {dt:<10.1f}')
    
    noise['net.state_dict()'] = net.state_dict()
    torch.save(noise, f'noise/noise_cifar10_resnet_pgd_train{name_text}.pt')
    # print

# ######## currently wont work with the rest of the code because not refactored and edited
# def val(epoch, net, loss_fn, data_loader, requires_control = True):
#     start_time = time.time()    
#     net.eval()

#     acc = []
#     loss = []
#     if requires_control:
#         orig_acc = []
#         orig_loss = []
#     for i, (orig, adv, label) in enumerate(data_loader):
#         adv = Variable(adv.cuda(non_blocking = True), volatile = True)
#         label = Variable(label.cuda(non_blocking = True), volatile = True)
#         logits = net(adv, defense = True)[-1]
#         acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
#         l = loss_fn(logits, label)
#         loss.append(l.data[0])
        
#         if requires_control:
#             orig = Variable(orig.cuda(non_blocking = True), volatile = True)
#             logits = net(orig)[-1]
#             orig_acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
#             l = loss_fn(logits, label)
#             orig_loss.append(l.data[0])            

#     acc = np.mean(acc)
#     loss = np.mean(loss)
#     if requires_control:
#         orig_acc = np.mean(orig_acc)
#         orig_loss = np.mean(orig_loss)
#     end_time = time.time()
#     dt = end_time - start_time

#     if requires_control:
#         print('Validation: loss %.5f, acc %.3f, orig_loss %.5f, orig_acc %.3f, time %3.1f' % (
#             loss, acc, orig_loss, orig_acc, dt))
#     else: 
#         print('Validation: loss %.5f, acc %.3f, time %3.1f' % (
#             loss, acc, dt))
#     # print
#     # print

 
# def test(net, data_loader, result_file_name, defense = True):
#     start_time = time.time()    
#     net.eval()

#     acc_by_attack = {}
#     for i, (adv, label, attacks) in enumerate(data_loader):
#         adv = Variable(adv.cuda(non_blocking = True), volatile = True)

#         adv_pred = net(adv, defense = defense)
#         _, idcs = adv_pred[-1].data.cpu().max(1)
#         corrects = idcs == label
#         for correct, attack in zip(corrects, attacks):
#             if acc_by_attack.has_key(attack):
#                 acc_by_attack[attack] += correct
#             else:
#                 acc_by_attack[attack] = correct
#     np.save(result_file_name,acc_by_attack)



##### refactored and edited this stuff
def test(epoch, net, loss_fn, data_loader, requires_control = True):
    ## log outputs to noise folder. store the latest epoch only
    noise = {
        'orig': [],
        'adv': [],
        'pred_noise': [],
        'actual_noise': [],
        'label': [],
    }


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
            noise['adv'].append(adv.detach().cpu())
            noise['label'].append(label.detach().cpu())
            noise['orig'].append(orig.detach().cpu())
            noise['pred_noise'].append(pred_noise.detach().cpu())
            noise['actual_noise'].append(actual_noise.detach().cpu())
            
            if requires_control:
                orig = Variable(orig.cuda(non_blocking = True), volatile = True)
                label = Variable(label.data, volatile = True)
                logits = net(orig)[-1]
                orig_acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
                l = loss_fn(logits, label)
                # orig_loss.append(l.data[0])
                orig_loss.append(l.item())
                l = l.detach()
                

    acc = np.mean(acc)
    loss = np.mean(loss)
    if requires_control:
        orig_acc = np.mean(orig_acc)
        orig_loss = np.mean(orig_loss)
    end_time = time.time()
    dt = end_time - start_time


            
    if requires_control:
        print(f'test : Epoch {epoch:<3}              : loss {loss:<7.5f}, acc {acc:<5.3f}, orig_loss {orig_loss:<7.5f}, orig_acc {orig_acc:<5.3f}, time {dt:<10.1f}')
    else: 
        print(f'test : Epoch {epoch:<3}              : loss {loss:<7.5f}, acc {acc:<5.3f}, time {dt:<10.1f}')
    
    torch.save(noise, f'noise/noise_cifar10_resnet_pgd_test{name_text}.pt')
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