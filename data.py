from keras.datasets import cifar10
from keras.utils import to_categorical


def get_cifar10_train_data():
    (train_x, train_y), _ = cifar10.load_data()

    train_y = to_categorical(train_y)
    train_x = train_x.astype('float32')
    train_x = train_x / 255.0

    return train_x, train_y


def get_cifar10_data():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    return train_x, train_y, test_x, test_y
