import torch
from torchsummary import summary 
from model import *
from PIL import Image
from torchvision import transforms as tf
from matplotlib import pyplot as plt
import numpy as np


path = 'noise/noise_cifar10_resnet_pgd_test.pt'
noise = torch.load(path)
orig         = noise['orig'][0][0]
adv          = noise['adv'][0][0]
pred_noise   = noise['pred_noise'][0][0]
actual_noise = noise['actual_noise'][0][0]
label        = noise['label'][0][0]
print(type(pred_noise[0]))

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[12, 4])
axs[0].imshow(tf.ToPILImage()(orig))
axs[1].imshow(tf.ToPILImage()(adv - pred_noise))
axs[2].imshow(tf.ToPILImage()(orig + actual_noise))
axs[0].set_title('original image')
axs[1].set_title('denoised image')
axs[2].set_title('adv image')
plt.show()