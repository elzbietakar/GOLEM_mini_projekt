import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        
        # changing first layer to match size of input pictures (32x32)
        # self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
           
        # updating the last linear layer
        self.resnet18.fc = nn.Linear(512, 10)
        
    def forward(self, x):
        return self.resnet18(x)


