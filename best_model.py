import torch.nn as nn
import torch.nn.functional as F
import torch

class ZuziaNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
             nn.Conv2d(3, 16, kernel_size=6, bias=False),
             nn.BatchNorm2d(16),
             nn.ReLU(),
        )
        self.block2 = nn.Sequential(
             nn.Conv2d(16, 64, kernel_size=3, bias=False),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout2d(p=0.3)
        )
        self.block3 = nn.Sequential(
             nn.Conv2d(64, 256, kernel_size=3, bias=False),
             nn.BatchNorm2d(256),
             nn.ReLU(),
             nn.Dropout2d(p=0.3),
        )
        self.block4 = nn.Sequential(
             nn.Conv2d(256, 64, kernel_size=3, bias=False),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.Dropout2d(p=0.2)
        )
        self.block5 = nn.Sequential(
             nn.Conv2d(64, 32, kernel_size=3, bias=False),
             nn.BatchNorm2d(32),
             nn.ReLU(),
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