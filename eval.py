import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import numpy as np


def eval(model, criterion, loader):
    mean_loss = 0.0
    counter = 0
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data in loader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            mean_loss += loss.item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            counter += 1
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy of the network on the 100 test images: {accuracy*100:.2f} %')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=np.nan)
    f1 = f1_score(y_true, y_pred, average='macro')
    classification_report = [accuracy, precision, recall, f1]
    return (mean_loss/counter, classification_report)