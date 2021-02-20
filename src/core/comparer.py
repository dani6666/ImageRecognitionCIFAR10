from src.core import trainer
from src.repository import cifar_dataset, neural_networks


def compare_all_neural_networks():
    train_x, train_y, test_x, test_y = cifar_dataset.get_cifar10_data()

    for network_name in neural_networks.networks_names:
        print("Testing " + network_name)

        network = trainer.get_and_train_network(network_name, train_x, train_y)

        _, accuracy = network.evaluate(test_x, test_y, verbose=0)

        print("Accuracy: " + str(accuracy))


def compare_trained_neural_networks():
    test_x, test_y = cifar_dataset.get_cifar10_test_data()

    for network_name in neural_networks.networks_names:

        network = trainer.get_network_if_trained(network_name)
        if network is None:
            continue

        print("Testing " + network_name)

        _, accuracy = network.evaluate(test_x, test_y, verbose=0)

        print("Accuracy: " + str(accuracy))
