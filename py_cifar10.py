import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


train_labels = []
train_data = np.empty((0, 3072))
test_labels = []
test_data = np.empty((0, 3072))


for i in range(1, 5):
    file = f'cifar-10-batches-py/data_batch_{i}'
    dictionary = unpickle(file)
    train_labels += dictionary[b'labels']
    train_data = np.concatenate([train_data, dictionary[b'data']])


file = f'cifar-10-batches-py/test_batch'
dictionary = unpickle(file)
test_labels = dictionary[b'labels']
test_data = dictionary[b'data']

for key in dictionary.keys():
    print(f'{key}   {type(dictionary[key])}')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = Cifar(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


