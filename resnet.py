import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from eval import eval
from train_epoch import train_epoch

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),  std=(0.247, 0.243, 0.261))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),  std=(0.247, 0.243, 0.261))
])

batch_size = 256

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in resnet.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = resnet.fc.in_features
print(num_ftrs)
resnet.fc = nn.Linear(num_ftrs, 10)

resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()

# Optimize only last
optimizer = optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

train_loss = []
eval_loss = []
metrics = []
epochs = []

how_many_epoch = 20
for epoch in range(how_many_epoch):
    tloss = train_epoch(resnet, criterion, optimizer, trainloader, epoch)
    eloss, classification_report = eval(resnet, criterion, testloader)

    train_loss.append(tloss)
    eval_loss.append(eloss)
    metrics.append(classification_report)
    epochs.append(epoch+1)

print(train_loss)
print(eval_loss)
print(metrics)

PATH = f"resnet_{how_many_epoch}epoch_Adam_{batch_size}"
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
torch.save(resnet.state_dict(),(f"{PATH}\model.pth"))