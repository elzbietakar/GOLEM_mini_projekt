import torch.nn as nn
import torch.nn.functional as F
import torch

#Given groups=1, weight of size [384, 24, 5, 5], expected input[64, 96, 15, 15] to have 24 channels, but got 96 channels instead
#Given groups=1, weight of size [384, 24, 5, 5], expected input[64, 96, 15, 15] to have 24 channels, but got 96 channels instead
class ZuziaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # self.conv1 = nn.Conv2d(3, 16, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(16, 32, 3)
        # self.fc1 = nn.Linear(32 * 3 * 3, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x