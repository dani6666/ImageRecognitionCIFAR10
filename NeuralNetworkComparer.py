from keras.datasets import cifar10
from keras.utils import to_categorical

from NeuralNetworksManager import NeuralNetworksManager


class NeuralNetworkComparer:

    @staticmethod
    def compare_neural_networks(networks_names):
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()

        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        train_x = train_x / 255.0
        test_x = test_x / 255.0

        for network_name in networks_names:
            print("Testing " + network_name)

            model = NeuralNetworksManager.get_trained_network(network_name, train_x, train_y)

            _, accuracy = model.evaluate(test_x, test_y, verbose=0)

            print("Accuracy: " + str(accuracy))

