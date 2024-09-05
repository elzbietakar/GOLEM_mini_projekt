import torch.nn as nn
import torch


class OlaNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


model = OlaNet4()
total_params = sum(p.numel() for p in model.parameters())
print("Amount of parameters:", total_params)