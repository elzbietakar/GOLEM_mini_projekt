import torch
from sklearn.metrics import classification_report


def eval(model, criterion, loader):
    correct = 0
    total = 0
    mean_loss = 0.0
    counter = 0
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data in loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            mean_loss += loss.item()
            y_true.extend(predicted.tolist())
            y_pred.extend(labels.tolist())
            counter += 1
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    return (mean_loss/counter, classification_report(y_true, y_pred))