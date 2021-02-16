import os

from keras import models

from AllNeuralNetworks import AllNeuralNetworks
from DataManager import DataManager


class NeuralNetworksManager:

    training_epochs = 50

    @staticmethod
    def delete_all_networks():
        if not os.path.exists("NeuralNetworkModels"):
            os.makedirs("NeuralNetworkModels")

        for filename in os.listdir("NeuralNetworkModels"):
            file_path = os.path.join("NeuralNetworkModels", filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    @staticmethod
    def retrain_networks(network_names):
        if not os.path.exists("NeuralNetworkModels"):
            os.makedirs("NeuralNetworkModels")

        for network_name in network_names:
            model_path = os.path.join("NeuralNetworkModels", network_name.replace(" ", "_"))
            if os.path.exists(model_path):
                os.remove(model_path)

        train_x, train_y = DataManager.get_cifar10_train_data()

        for network_name in network_names:
            model_path = os.path.join("NeuralNetworkModels", network_name.replace(" ", "_"))

            model = AllNeuralNetworks.get_network_model(network_name)

            print("Retraining model: " + network_name)
            augmentation_mode = AllNeuralNetworks.get_augmentation_mode(network_name)
            train_x, train_y = NeuralNetworksManager.augment_data(augmentation_mode, train_x, train_y)

            model.fit(train_x, train_y, epochs=NeuralNetworksManager.training_epochs, verbose=2)

            model.save(model_path)

    @staticmethod
    def train_all_networks(network_names):
        if not os.path.exists("NeuralNetworkModels"):
            os.makedirs("NeuralNetworkModels")

        train_x, train_y = DataManager.get_cifar10_train_data()

        for network_name in network_names:
            model_path = os.path.join("NeuralNetworkModels", network_name.replace(" ", "_"))

            if not os.path.exists(model_path):
                model = AllNeuralNetworks.get_network_model(network_name)

                print("Training new model: " + network_name)
                augmentation_mode = AllNeuralNetworks.get_augmentation_mode(network_name)
                train_x, train_y = NeuralNetworksManager.augment_data(augmentation_mode, train_x, train_y)

                model.fit(train_x, train_y, epochs=NeuralNetworksManager.training_epochs, verbose=2)

                model.save(model_path)

    @staticmethod
    def get_trained_network(network_name, train_x, train_y):

        if not os.path.exists("NeuralNetworkModels"):
            os.makedirs("NeuralNetworkModels")

        model_path = os.path.join("NeuralNetworkModels", network_name.replace(" ", "_"))

        try:
            saved_model = models.load_model(model_path)
            return saved_model
        except OSError:
            model = AllNeuralNetworks.get_network_model(network_name)

            print("Training new model: " + network_name)
            augmentation_mode = AllNeuralNetworks.get_augmentation_mode(network_name)
            train_x, train_y = NeuralNetworksManager.augment_data(augmentation_mode, train_x, train_y)

            model.fit(train_x, train_y, epochs=NeuralNetworksManager.training_epochs, verbose=0)

            model.save(model_path)
            return model

    @staticmethod
    def augment_data(image_generator, train_x, train_y):
        if image_generator is None:
            return train_x, train_y

        return image_generator.flow(train_x, train_y), None
