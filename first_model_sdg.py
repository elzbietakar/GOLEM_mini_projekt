import torch
import torchvision
import torchvision.transforms as transforms
from eval import eval
import torch.nn as nn
import torch.optim as optim
from train_epoch import train_epoch
from model1 import ZuziaNet
from model2 import ZuziaNet2
from model3 import ZuziaNet3
import matplotlib.pyplot as plt
import os

train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(degrees=30),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),  std=(0.247, 0.243, 0.261))
    ])
test_transform = transforms.Compose([
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

how_many_epoch = 2
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

PATH = f"znet2_{how_many_epoch}epoch_Adam_{batch_size}"
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
precision = [report[1] for report in metrics]
recall = [report[2] for report in metrics]
f1 = [report[3] for report in metrics]
plt.plot(epochs, accuracy, label='Accuracy')
plt.plot(epochs, precision, label='Precision')
plt.plot(epochs, recall, label='Recall')
plt.plot(epochs, f1, label='F1 score')

plt.title('Plot compares accuracy, precision, recall, f1 on every epoch')
plt.xlabel('Epochs')
plt.ylabel('Metrics')

plt.legend()

plt.savefig(f"{PATH}\plot_metrics.png")
plt.show()

#Saves model
torch.save(znet.state_dict(),(f"{PATH}\model.pth"))