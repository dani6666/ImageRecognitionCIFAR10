from src.core import trainer
from src.repository import cifar_dataset, neural_networks


def compare_all_neural_networks():
    train_x, train_y, test_x, test_y = cifar_dataset.get_cifar10_data()

    for network_name in neural_networks.networks_names:
        print("Testing " + network_name)

        model = trainer.get_trained_network(network_name, train_x, train_y)

        _, accuracy = model.evaluate(test_x, test_y, verbose=0)

        print("Accuracy: " + str(accuracy))
