import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from cnn import NeuralNet
from torch.optim import SGD
from torch.nn import CrossEntropyLoss




def transform_data(mean_val, std_val):
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (mean_val), std = (std_val))

])
    return transformation