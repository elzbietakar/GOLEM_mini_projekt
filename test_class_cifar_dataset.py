from py_cifar10 import train_labels, train_data
from class_cifar_dataset import CifarDataset
import numpy as np

def test_cifar_dataset():
    assert train_labels[0] == 6
    assert np.all(train_data[0][:9] == np.array([ 59,  43,  50,  68,  98, 119, 139, 145, 149]))
    cifardataset = CifarDataset(train_labels[0], train_data[0])
    assert cifardataset.labels == 6
    assert np.all(cifardataset.data[:9] == np.array([ 59,  43,  50,  68,  98, 119, 139, 145, 149]))

