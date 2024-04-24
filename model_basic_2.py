import torch.nn as nn
import torch


class OlaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # 32 -3 +1 = 30
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # 30 - 2 = 28 -2/2 +1 = 14
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # 14 - 2 = 12 -2/2 +1 = 6

        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x