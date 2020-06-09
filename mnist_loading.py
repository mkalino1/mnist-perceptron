import numpy as np


def load_data():

    with open('train-images.idx3-ubyte') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)[16:]
        training_images = loaded.reshape((60000, 784, 1)).astype(np.float)
        training_images = training_images/255

    with open('train-labels.idx1-ubyte') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)
        training_labels = loaded[8:].reshape((60000,)).astype(np.int)

    with open('t10k-images.idx3-ubyte', 'rb') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)[16:]
        test_images = loaded.reshape((10000, 784, 1)).astype(np.float)
        test_images = test_images/255

    with open('t10k-labels.idx1-ubyte', 'rb') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)
        test_labels = loaded[8:].reshape((10000,)).astype(np.int)

    # Dane zgrupowane w listę par (image, label)
    training_data = list(zip(training_images, training_labels))
    test_data = list(zip(test_images, test_labels))

    return training_data, test_data


