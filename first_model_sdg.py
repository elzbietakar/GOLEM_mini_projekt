import torch
import torchvision
import torchvision.transforms as transforms
from eval import eval
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from train_epoch import train_epoch
from model1 import ZuziaNet
#from NeuralNetworkModel import Net
import matplotlib.pyplot as plt

# Transform our data to tensor and tranform every value [0, 1]
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.4),
        #transforms.ColorJitter(0.3, 0.2, 0.2, 0.05),
        transforms.RandomRotation(degrees=30),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),  std=(0.247, 0.243, 0.261))
    ])
test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),  std=(0.247, 0.243, 0.261))
    ])


# How many pictures are analyzed in one iteration
batch_size = 64

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



znet = ZuziaNet()
#znet = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(znet.parameters(), lr=0.001, momentum=0.9)

train_loss = []
eval_loss = []
metrics = []
epochs = []

for epoch in range(4):
    tloss = train_epoch(znet, criterion, optimizer, trainloader, epoch)
    eloss, classification_report = eval(znet, criterion, testloader)

    train_loss.append(tloss)
    eval_loss.append(eloss)
    metrics.append(classification_report)
    epochs.append(epoch+1)

print(train_loss)
print(eval_loss)
print(metrics)

# Wykres z dwoma zmiennymi jako punkty
plt.scatter(epochs, train_loss, label='Training loss')
plt.scatter(epochs, eval_loss, label='Evaluation loss')

plt.title('Plot compares training and evaluation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()

plt.savefig('zuzianet_100epochsSGD_64b.png')
plt.show()
