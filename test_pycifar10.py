from py_cifar10 import train_labels, train_data, test_labels, test_data
import numpy as np

def test_importing_train_labels():
    assert train_labels[0] == 6
    assert np.all(train_data[0][:9] == np.array([ 59,  43,  50,  68,  98, 119, 139, 145, 149]))

def test_importing_test_labels():
    assert test_labels[0] == 3
    assert np.all(test_data[0][:3] == np.array([ 158,  159,  165]))

