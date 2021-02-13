import os

from keras import models

from AllNeuralNetworks import AllNeuralNetworks


class NeuralNetworksManager:

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
            model.fit(train_x, train_y, epochs=20, batch_size=64, validation_data=(train_x, train_y), verbose=0)

            model.save(model_path)
            return model
