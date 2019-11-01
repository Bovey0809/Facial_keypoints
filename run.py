from data_load import Rescale, RandomCrop, Normalize, ToTensor
from data_load import FacialKeypointsDataset
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from models import Net
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchsnooper
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda')
net = Net()

data_transform = transforms.Compose(
    [Rescale(256),
     RandomCrop(224),
     Normalize(),
     ToTensor()]
)

train_dataset = FacialKeypointsDataset(
    "./data/training_frames_keypoints.csv",
    "./data/training",
    data_transform
)

test_dataset = FacialKeypointsDataset(
    "./data/test_frames_keypoints.csv",
    "./data/test",
    data_transform
)

batch_size = 256
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

print(net)
