import os

from keras import models

from AllNeuralNetworks import AllNeuralNetworks
from DataManager import DataManager


class NeuralNetworksManager:

    training_epochs = 50

    @staticmethod
    def train_all_networks(network_names):
        if not os.path.exists("NeuralNetworkModels"):
            os.makedirs("NeuralNetworkModels")

        train_x, train_y = DataManager.get_cifar10_train_data()

        for network_name in network_names:
            model_path = 'NeuralNetworkModels/' + network_name.replace(" ", "_")

            if not os.path.exists(model_path):
                model = AllNeuralNetworks.get_network_model(network_name)

                print("Training new model: " + network_name)
                model.fit(train_x, train_y, epochs=NeuralNetworksManager.training_epochs,
                          batch_size=64, validation_data=(train_x, train_y), verbose=0)

                model.save(model_path)

    @staticmethod
    def get_trained_network(network_name, train_x, train_y):

        if not os.path.exists("NeuralNetworkModels"):
            os.makedirs("NeuralNetworkModels")

        model_path = 'NeuralNetworkModels/' + network_name.replace(" ", "_")

        try:
            saved_model = models.load_model(model_path)
            return saved_model
        except OSError:
            model = AllNeuralNetworks.get_network_model(network_name)

            print("Training new model: " + network_name)
            model.fit(train_x, train_y, epochs=NeuralNetworksManager.training_epochs,
                      batch_size=64, validation_data=(train_x, train_y), verbose=0)

            model.save(model_path)
            return model
