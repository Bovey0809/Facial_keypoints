import torch
import torch.nn as nn


class Alexnet(nn.Module):

    def __init__(self):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=3),  # 55x55
            nn.ReLU(True),  # 27x27
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),  # 13x13
            nn.ReLU(True),  # 13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=2),  # 13x13
            nn.ReLU(True),  # 13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=2),  # 13x13
            nn.ReLU(True),  # 13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=2),  # 13x13
            nn.ReLU(True),  # 13x13
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6
        )
        # avgpool is from another paper.
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classfier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 68*2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classfier(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 7, 3, 1),
            nn.SELU(True),
            nn.Conv2d(32, 64, 5, 3, 0),
            nn.SELU(True),
            nn.Conv2d(64, 128, 5, 3, 1),
            nn.SELU(True),
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.SELU(True),
            nn.Conv2d(256, 512, 3, 1, 0),
            nn.SELU(True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.SELU(True)
        )
        self.classifer = nn.Sequential(
            nn.Linear(4*4*512, 1024),
            nn.SELU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.SELU(True),
            nn.Dropout(),
            nn.Linear(1024, 136)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x
