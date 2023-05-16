import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import os
from os.path import expanduser
import collections
from torch.autograd import Variable
from resnext_features import resnext101_32x4d_features
from resnext_features import resnext101_64x4d_features
import torchvision

 
class Conv(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, stride = 1, expansion = 4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(n_out)
        self.conv3 = nn.Conv2d(n_out, n_out * expansion, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(n_out * expansion)

        self.downsample = None
        if stride != 1 or n_in != n_out * expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(n_in, n_out * expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(n_out * expansion))

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x


# net_type = 'resnext101'
# config['net_type'] = net_type
# input_size = [299, 299]
# block = Conv
# fwd_out = [64, 128, 256, 256, 256]
# num_fwd = [2, 3, 3, 3, 3]
# back_out = [64, 128, 256, 256]
# num_back = [2, 3, 3, 3]
# n = 1
# hard_mining = 0
# loss_norm = False

class Denoise(nn.Module):
    # Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back)
    # Denoise(
    #     h_in     = 299, 
    #     w_in     = 299, 
    #     block    = Conv, same as: Sequential(Conv2d, BatchNorm2d, ReLU), 
    #     fwd_in   = 3, 
    #     fwd_out  = [64, 128, 256, 256, 256], 
    #     num_fwd  = [2, 3, 3, 3, 3], 
    #     back_out = [64, 128, 256, 256], 
    #     num_back = [2, 3, 3, 3]
    # )
    def __init__(self, h_in, w_in, block, fwd_in, fwd_out, num_fwd, back_out, num_back):
        super(Denoise, self).__init__()

        h, w = [], []
        for i in range(len(num_fwd)):
            h.append(h_in)
            w.append(w_in)
            h_in = int(np.ceil(float(h_in) / 2))
            w_in = int(np.ceil(float(w_in) / 2))

        if block is Bottleneck:
            expansion = 4
        else:
            expansion = 1
        
        fwd = []
        n_in = fwd_in
        for i in range(len(num_fwd)):
            group = []
            for j in range(num_fwd[i]):
                if j == 0:
                    if i == 0:
                        stride = 1
                    else:
                        stride = 2
                    group.append(block(n_in, fwd_out[i], stride = stride))
                else:
                    group.append(block(fwd_out[i] * expansion, fwd_out[i]))
            n_in = fwd_out[i] * expansion
            fwd.append(nn.Sequential(*group))
        self.fwd = nn.ModuleList(fwd)

        upsample = []
        back = []
        n_in = (fwd_out[-2] + fwd_out[-1]) * expansion
        for i in range(len(num_back) - 1, -1, -1):
            upsample.insert(0, nn.Upsample(size = (h[i], w[i]), mode = 'bilinear'))
            group = []
            for j in range(num_back[i]):
                if j == 0:
                    group.append(block(n_in, back_out[i]))
                else:
                    group.append(block(back_out[i] * expansion, back_out[i]))
            if i != 0:
                n_in = (back_out[i] + fwd_out[i - 1]) * expansion
            back.insert(0, nn.Sequential(*group))
        self.upsample = nn.ModuleList(upsample)
        self.back = nn.ModuleList(back)

        self.final = nn.Conv2d(back_out[0] * expansion, fwd_in, kernel_size = 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        outputs = []
        for i in range(len(self.fwd)):
            out = self.fwd[i](out)
            if i != len(self.fwd) - 1:
                outputs.append(out)
        
        for i in range(len(self.back) - 1, -1, -1):
            out = self.upsample[i](out)
            out = torch.cat((out, outputs[i]), 1)
            out = self.back[i](out)
        out = self.final(out)
        out = self.sigmoid(out)
        out = out.clone() + x
        return out


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, denoise, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.features = resnext101_64x4d_features
        self.fc = nn.Linear(2048, num_classes)

        self.denoise = denoise

    def forward(self, x, defense = False):
        outputs = []
        if defense:
            x = self.denoise(x)
            # noise = self.denoise(x)
            # x = x - noise
        outputs.append(x)
        x = self.features(x)
        outputs.append(x)
        size = x.size()
        x = x.view(size[0], size[1], -1)
        x = x.mean(2)
        x = self.fc(x)
        outputs.append(x)
        return outputs


# ishaan: new class added to use the resnet we pretrained as the victim model.
class Ishaan_Resnet(nn.Module):
    def __init__(self, denoise):
        super().__init__()
        resnet = torchvision.models.resnet18(weights="DEFAULT")
        resnet.fc = nn.Linear(512, 10, bias=True)
        resnet.load_state_dict(torch.load("dataset/pgd_samples_CIFAR10.pt")['model'])
        self.resnet = resnet
        self.denoise = denoise

        
        self.denoise_fn = nn.Sequential(
            torchvision.transforms.Resize(self.denoise.upsample[0].size),
            self.denoise,
            torchvision.transforms.Resize([32, 32]),
        )
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,

            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
        )


    def forward(self, x, defense = False):
        outputs = []
        if defense:
            x = self.denoise_fn(x)
        outputs.append(x)
        x = self.features(x)
        outputs.append(x)
        size = x.size()
        x = x.view(size[0], size[1], -1)
        x = x.mean(2)
        # x = self.fc(x)
        x = self.resnet.fc(x)
        outputs.append(x)

        return outputs



class DenoiseLoss(nn.Module):
    def __init__(self, n, hard_mining = 0, norm = False):
        super(DenoiseLoss, self).__init__()
        self.n = n
        assert(hard_mining >= 0 and hard_mining <= 1)
        self.hard_mining = hard_mining
        self.norm = norm

    def forward(self, x, y):
        loss = torch.pow(torch.abs(x - y), self.n) / self.n
        if self.hard_mining > 0:
            loss = loss.view(-1)
            k = int(loss.size(0) * self.hard_mining)
            loss, idcs = torch.topk(loss, k)
            y = y.view(-1)[idcs]

        loss = loss.mean()
        if self.norm:
            norm = torch.pow(torch.abs(y), self.n)
            norm = norm.data.mean()
            loss = loss / norm
        return loss


class Loss(nn.Module):
    def __init__(self, n, hard_mining = 0, norm = False):
        super(Loss, self).__init__()
        self.loss = DenoiseLoss(n, hard_mining, norm)

    def forward(self, x, y):
        z = []
        for i in range(len(x)):
            z.append(self.loss(x[i], y[i]))
        return z
    

# ishaan: code for this class changed to use the resnet.
class Net(nn.Module):
    def __init__(self, input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining = 0, loss_norm = False):
        super(Net, self).__init__()
        denoise = Denoise(input_size[0], input_size[1], block, 3, fwd_out, num_fwd, back_out, num_back)

        # self.net = ResNeXt101_64x4d(denoise)   # ishaan: code changed to load our pretrained model.
        self.net = Ishaan_Resnet(denoise)
        self.loss = Loss(n, hard_mining, loss_norm)

    def forward(self, orig_x, adv_x, requires_control = True, train = True):
        orig_outputs = self.net(orig_x)

        if requires_control:
            control_outputs = self.net(adv_x)
            control_loss = self.loss(control_outputs, orig_outputs)

        if train:
            adv_x.volatile = False
            for i in range(len(orig_outputs)):
                orig_outputs[i].volatile = False
        adv_outputs = self.net(adv_x, defense = True)
        loss = self.loss(adv_outputs, orig_outputs)

        if not requires_control:
            return orig_outputs[-1], adv_outputs[-1], loss
        else:
            return orig_outputs[-1], adv_outputs[-1], loss, control_outputs[-1], control_loss


def get_net(input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining = 0, loss_norm = False):
    net = Net(input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining, loss_norm)

    # ishaan: code added in the Net function is for loading the pretrained model on PGD samples that we trained

    return net