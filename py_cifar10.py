import numpy as np
from class_cifar_dataset import CifarDataset
from torchvision import transforms

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
train_data = np.empty((0, 3072))
test_labels = []
test_data = np.empty((0, 3072))


for i in range(1, 6):
    file = f'cifar-10-batches-py/data_batch_{i}'
    dictionary = unpickle(file)
    print(dictionary[b'data'].shape)
    train_labels += dictionary[b'labels']
    train_data = np.concatenate([train_data, dictionary[b'data']])
    #print(type(train_data))
    #print(type( dictionary[b'data']))


file = f'cifar-10-batches-py/test_batch'
dictionary = unpickle(file)
test_labels = dictionary[b'labels']
test_data = dictionary[b'data']

train_data = train_data.reshape(len(train_data),3, 32, 32)
train_data = train_data.transpose(0, 2, 3, 1)
test_data = test_data.reshape(len(test_data),3, 32, 32)
test_data = test_data.transpose(0, 2, 3, 1)

for key in dictionary.keys():
    print(f'{key}   {type(dictionary[key])}')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = CifarDataset(train_labels, train_data, transform)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                          shuffle=True, num_workers=2)

testset = CifarDataset(test_labels, test_data, transform)

#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("typ")
trainset.print_summary()
one = trainset.get_one(0)
one_data = one[0].reshape(3, 32, 32)
one_data = one_data.transpose(1, 2, 0)
print(one_data.ndim)
print(type(one))
print(type(one[0]))
print(type(one[1]))
data_one_transformed = transform(one_data)
trainset.print_types(0)
print(trainset.__getitem__(0)[0])
