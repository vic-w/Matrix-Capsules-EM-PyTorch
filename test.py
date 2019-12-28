#coding=utf-8

from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from PIL import Image

from model import capsules

transform=transforms.Compose([ transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

A=64
B=8
C=16
D=16
E=10
model = capsules(A=A, B=B, C=C, D=D, E=E, 
                     iters=2).cuda()
model.load_state_dict(torch.load('snapshots/model_10.pth'))
model.eval()

img = transform(Image.open('bmp/test_1.bmp')).reshape([1,1,28,28]).cuda()
pos, result = model(img)


print(np.argmax(result.cpu().detach().numpy()))
