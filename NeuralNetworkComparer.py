from keras.datasets import cifar10
from keras.utils import to_categorical

from DataManager import DataManager
from NeuralNetworksManager import NeuralNetworksManager


class NeuralNetworkComparer:

    @staticmethod
    def compare_neural_networks(networks_names):
        train_x, train_y, test_x, test_y = DataManager.get_cifar10_data()

        for network_name in networks_names:
            print("Testing " + network_name)

            model = NeuralNetworksManager.get_trained_network(network_name, train_x, train_y)

            _, accuracy = model.evaluate(test_x, test_y, verbose=0)

            print("Accuracy: " + str(accuracy))

