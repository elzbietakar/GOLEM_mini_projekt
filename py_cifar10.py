import numpy as np
from class_cifar_dataset import CifarDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def reshape(data):
    data = data.reshape(len(data),3, 32, 32)
    data = data.transpose(0, 2, 3, 1)
    return data

train_labels = []
train_data = np.empty((0, 3072), dtype=np.uint8)
test_labels = []
test_data = np.empty((0, 3072), dtype=np.uint8)


for i in range(1, 6):
    file = f'cifar-10-batches-py/data_batch_{i}'
    dictionary = unpickle(file)
    print(dictionary[b'data'].shape)
    train_labels += dictionary[b'labels']
    train_data = np.concatenate([train_data, dictionary[b'data']])
    print("typ pustego")
    print(type(train_data[0][0]))
    # print(type(train_data))
    #print(type( dictionary[b'data']))


file = f'cifar-10-batches-py/test_batch'
dictionary = unpickle(file)
test_labels = dictionary[b'labels']
test_data = dictionary[b'data']

train_data = train_data.reshape(len(train_data), 3, 32, 32)
train_data = train_data.transpose(0, 2, 3, 1)
test_data = test_data.reshape(len(test_data), 3, 32, 32)
test_data = test_data.transpose(0, 2, 3, 1)

print(type(train_data[0][0][0][0]))
print(train_data[0])



for key in dictionary.keys():
    print(f'{key}   {type(dictionary[key])}')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = CifarDataset(train_labels, train_data, transform)
train_dataloader = DataLoader(trainset, batch_size=128, shuffle=True)


testset = CifarDataset(test_labels, test_data, transform)
test_dataloader = DataLoader(testset, batch_size=128, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print("typ")

print(trainset.__getitem__(0)[0])
