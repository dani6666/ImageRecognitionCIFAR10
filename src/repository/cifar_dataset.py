from keras.datasets import cifar10
from keras.utils import to_categorical


def get_cifar10_train_data():
    (train_x, train_y), _ = cifar10.load_data()

    train_y = to_categorical(train_y)
    train_x = prepare_x(train_x)

    return train_x, train_y


def get_cifar10_test_data():
    _, (test_x, test_y) = cifar10.load_data()

    test_y = to_categorical(test_y)
    test_x = prepare_x(test_x)

    return test_x, test_y


def get_cifar10_data():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    train_x = prepare_x(train_x)
    test_x = prepare_x(test_x)

    return train_x, train_y, test_x, test_y


def prepare_x(x):
    return x.astype('float32') / 255.0
