import trainer
import data


def compare_neural_networks(networks_names):
    train_x, train_y, test_x, test_y = data.get_cifar10_data()

    for network_name in networks_names:
        print("Testing " + network_name)

        model = trainer.get_trained_network(network_name, train_x, train_y)

        _, accuracy = model.evaluate(test_x, test_y, verbose=0)

        print("Accuracy: " + str(accuracy))
