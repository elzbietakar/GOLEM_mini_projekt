import torch.nn as nn
import torch.nn.functional as F
import torch

class ZuziaNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
             nn.Conv2d(3, 32, kernel_size=7, bias=False),
             nn.BatchNorm2d(32),
             nn.ReLU(),
             nn.Dropout2d(p=0.1),
        )
        self.block2 = nn.Sequential(
             nn.Conv2d(32, 128, kernel_size=3, bias=False),
             nn.BatchNorm2d(128),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout2d(p=0.1)
        )
        self.block3 = nn.Sequential(
             nn.Conv2d(128,512, kernel_size=3, bias=False),
             nn.BatchNorm2d(512),
             nn.ReLU(),
             nn.Dropout2d(p=0.1),
        )
        self.block4 = nn.Sequential(
             nn.Conv2d(512, 128, kernel_size=3, bias=False),
             nn.BatchNorm2d(128),
             nn.ReLU(),
             nn.Dropout2d(p=0.1)
        )
        self.block5 = nn.Sequential(
             nn.Conv2d(128, 32, kernel_size=3, bias=False),
             nn.BatchNorm2d(32),
             nn.ReLU(),
             nn.Dropout2d(p=0.1),
        )
        self.fc1 = nn.Linear(32 * 6 * 6, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x