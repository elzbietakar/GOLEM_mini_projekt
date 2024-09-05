from small_model import OlaNet4
from resnet_model import ResNet
from big_model_with_dropout import ZuziaNet
from best_model import ZuziaNet2
onet = OlaNet4()
znet = ZuziaNet()
znet2 = ZuziaNet2()
resnet = ResNet()

def parameters(model):
    params = sum(p.numel() for p in model.parameters())
    print(params)

parameters(onet)
parameters(znet)
parameters(znet2)
parameters(resnet)
