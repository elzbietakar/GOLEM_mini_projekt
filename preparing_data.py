import numpy as np


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