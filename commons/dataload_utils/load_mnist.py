from .read_idx import read_idx
from . import np


def load_mnist():
    X_train = read_idx('../datasets/train_mnist_images.idx3-ubyte')
    X_train = np.expand_dims(X_train, axis=-1)
    Y_train = read_idx('../datasets/train_mnist_labels.idx1-ubyte')
    Y_train = np.expand_dims(Y_train, axis=-1)
    X_test = read_idx('../datasets/test_mnist_images.idx3-ubyte')
    X_test = np.expand_dims(X_test, axis=-1)
    Y_test = read_idx('../datasets/test_mnist_labels.idx1-ubyte')
    Y_test = np.expand_dims(Y_test, axis=-1)

    return X_train, Y_train, X_test, Y_test
