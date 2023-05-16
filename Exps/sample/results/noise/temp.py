import torch
from resnet import Net, Conv, get_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


"""Sample Pytorch defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""


import os
import argparse
import math
import numpy as np

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from resnext101 import get_model as get_model4

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR', default='',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE', default='',
                    help='Output file to save labels.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--checkpoint_path2', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor
    
    

    # args = parser.parse_args()

    # if not os.path.exists(args.input_dir):
    #     print("Error: Invalid input folder %s" % args.input_dir)
    #     exit(-1)
    # if not args.output_file:
    #     print("Error: Please specify an output file")
    #     exit(-1)
        
tf = transforms.Compose([
        transforms.Resize([299,299]),
        transforms.ToTensor()
])

mean_torch = autograd.Variable(torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1,3,1,1]).astype('float32')).to(device), volatile=True)
std_torch = autograd.Variable(torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1,3,1,1]).astype('float32')).to(device), volatile=True)
mean_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).to(device), volatile=True)
std_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).to(device), volatile=True)


config, net = get_model4()



orig_x = torch.Tensor(7, 3, 299, 299).to(device)
adv_x  = torch.Tensor(7, 3, 299, 299).to(device)

output = net(orig_x, adv_x)

torch.save(output, 'sample_output.pt')

