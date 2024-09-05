import torch.nn as nn
import torch

class ElaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
             nn.Conv2d(3, 32, kernel_size=5, bias=False),
             nn.BatchNorm2d(32),
             nn.ReLU(),
             nn.MaxPool2d(2,2),
             nn.Dropout(0.1)
        )
        #wyjście: (32 - 5 +1) / 2 = 14

        self.block2 = nn.Sequential(
             nn.Conv2d(32, 64, kernel_size=3, bias=False),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(0.2)
        )
        #wyjście: (14 - 3 +1) / 2 = 6

        self.block3 = nn.Sequential(
             nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
             nn.BatchNorm2d(96),
             nn.ReLU(),
             nn.Dropout(0.2),
        )
        #wyjście: 6 -3 +2 +1 = 6

        self.block4 = nn.Sequential(
             nn.Conv2d(96, 64, kernel_size=3, bias=False),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.Dropout(0.1)
        )
        #wyjście: 6 -3 +1 = 4

        self.fc1 = nn.Linear(64 * 4 * 4, 216)
        self.fc2 = nn.Linear(216, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x