from resnet_model import ResNet
from model1 import ZuziaNet
from model2 import ZuziaNet2
from model3 import ZuziaNet3
znet = ZuziaNet()
znet2 = ZuziaNet2()
znet3 = ZuziaNet3()
resnet = ResNet()

def parameters(model):
    params = sum(p.numel() for p in model.parameters())
    print(params)

parameters(znet)
parameters(resnet)
parameters(znet2)
parameters(znet3)