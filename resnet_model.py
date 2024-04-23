import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        
        # Zmiana pierwszej warstwy
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
        for name, param in self.resnet18.named_parameters():
            if 'conv1' in name or 'bn1' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
           
        
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True  
        

        # Zmiana liczby klas
        self.resnet18.fc = nn.Linear(512, 10)
        
    def forward(self, x):
        return self.resnet18(x)

resnet = ResNet()

input_data = torch.randn(1, 3, 32, 32) 
output = resnet(input_data)
print(output.shape)
