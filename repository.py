import os

from keras import models


if not os.path.exists("NeuralNetworkModels"):
    os.makedirs("NeuralNetworkModels")


def get_network(name):
    path = get_path(name)
    try:
        saved_model = models.load_model(path)
        return saved_model
    except OSError:
        return None


def save_network(network, name):
    path = get_path(name)
    network.save(path)


def network_exists(name):
    path = get_path(name)
    return os.path.exists(path)


def delete_network(name):
    path = get_path(name)
    if os.path.exists(path):
        os.remove(path)


def delete_all_networks():
    for filename in os.listdir("NeuralNetworkModels"):
        file_path = get_path(filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def get_path(network_name):
    return os.path.join("NeuralNetworkModels", network_name.replace(" ", "_"))
