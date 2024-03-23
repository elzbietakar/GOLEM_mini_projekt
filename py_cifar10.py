
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    file = 'cifar-10-batches-py/data_batch_1'

    dictionary_1 = unpickle(file)

    print(dictionary_1)

main()