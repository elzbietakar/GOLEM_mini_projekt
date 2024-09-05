import torch
import torchvision
import torchvision.transforms as transforms
from eval import eval
import torch.nn as nn
import torch.optim as optim
from train_epoch import train_epoch
from big_model_with_dropout import ZuziaNet
from model2_v3 import ZuziaNet2
from model3 import ZuziaNet3
import matplotlib.pyplot as plt
import os

train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.1),
        transforms.ColorJitter(brightness = 0.1,contrast = 0.1 ,saturation =0.1 ),
        transforms.RandomAdjustSharpness(sharpness_factor = 2, p = 0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),  std=(0.247, 0.243, 0.261))
    ])

#transforms.RandomHorizontalFlip(p=0.4),
# transforms.RandomRotation(degrees=30),

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),  std=(0.247, 0.243, 0.261))
    ])


# How many pictures are analyzed in one iteration
batch_size = 256

# Defining datasets and dataloaders for train and test data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# Our classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



znet = ZuziaNet2()

if torch.cuda.is_available():
    znet.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(znet.parameters(), lr=0.001)
# optimizer = optim.SGD(znet.parameters(), lr=0.001, momentum=0.9)

train_loss = []
eval_loss = []
metrics = []
epochs = []

how_many_epoch = 50
for epoch in range(how_many_epoch):
    tloss = train_epoch(znet, criterion, optimizer, trainloader, epoch)
    eloss, classification_report = eval(znet, criterion, testloader)

    train_loss.append(tloss)
    eval_loss.append(eloss)
    metrics.append(classification_report)
    epochs.append(epoch+1)

print(train_loss)
print(eval_loss)
print(metrics)

PATH = f"zresnet_fine_tuning_224_{how_many_epoch}epoch_SGD_{batch_size}"
if not os.path.exists(PATH):
    os.makedirs(PATH)

plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, eval_loss, label='Evaluation loss')

plt.title('Plot compares training and evaluation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()

plt.savefig(f"{PATH}\plot_loss.png")
plt.show()

accuracy = [report[0] for report in metrics]
plt.plot(epochs, accuracy, label='Accuracy')

plt.title('Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.grid(True)
plt.legend()

plt.savefig(f"{PATH}\plot_metrics.png")
plt.show()

#Saves model
torch.save(znet.state_dict(),(f"{PATH}\model.pth"))