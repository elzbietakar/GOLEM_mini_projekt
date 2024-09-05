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
        )
        # (28
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 26
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        # 26 - 2 = 24 -2 /2 +1 = 12
        self.block5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        # 12 - 2 = 10 -2 /2 +1 = 5

        self.fc1 = nn.Linear(16 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x