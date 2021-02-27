import os

import jsonpickle
from keras import models


networks_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join("storage", "history"))

if not os.path.exists(networks_path):
    os.makedirs(networks_path)


def get_learning_history(network_name):
    path = get_path(network_name)
    if not os.path.exists(path):
        return None

    file = open(path)
    data = file.read()
    file.close()
    return jsonpickle.loads(data)


def save_learning_history(history, network_name):
    path = get_path(network_name)
    data = jsonpickle.dumps(history)
    file = open(path, mode="w")
    file.write(data)
    file.close()


def network_learning_history_exists(network_name):
    path = get_path(network_name)
    return os.path.exists(path)


def delete_learning_history(network_name):
    path = get_path(network_name)
    if os.path.exists(path):
        os.remove(path)


def delete_all_learning_history():
    for filename in os.listdir(networks_path):
        file_path = get_path(filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def get_path(network_name):
    return os.path.join(networks_path, network_name.replace(" ", "_"))
