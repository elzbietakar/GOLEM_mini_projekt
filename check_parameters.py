from resnet_model import resnet
from model1_e import ElaNet
enet= ElaNet()

resnet_params = sum(p.numel() for p in resnet.parameters())
print(resnet_params)

enet_params = sum(p.numel() for p in enet.parameters())
print(enet_params)