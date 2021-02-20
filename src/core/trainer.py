from src.repository import cifar_dataset
from src.repository import neural_networks, networks_repository

training_epochs = 50


def retrain_networks(network_names):

    for network_name in network_names:
        networks_repository.delete_network(network_name)

    train_x, train_y = cifar_dataset.get_cifar10_train_data()

    for network_name in network_names:
        model = neural_networks.get_network_model(network_name)

        print("Retraining network: " + network_name)
        augmentation_mode = neural_networks.get_augmentation_mode(network_name)
        train_x, train_y = augment_data(augmentation_mode, train_x, train_y)

        model.fit(train_x, train_y, epochs=training_epochs, verbose=2)

        networks_repository.save_network(model, network_name)


def delete_all_trained_networks():
    networks_repository.delete_all_networks()


def train_all_networks():

    train_x, train_y = cifar_dataset.get_cifar10_train_data()

    for network_name in neural_networks.networks_names:
        if not networks_repository.network_exists(network_name):
            model = neural_networks.get_network_model(network_name)

            print("Training new network: " + network_name)
            augmentation_mode = neural_networks.get_augmentation_mode(network_name)
            train_x, train_y = augment_data(augmentation_mode, train_x, train_y)

            model.fit(train_x, train_y, epochs=training_epochs, verbose=2)

            networks_repository.save_network(model, network_name)


def get_trained_network(network_name, train_x, train_y):

    saved_network = networks_repository.get_network(network_name)
    if saved_network is not None:
        return saved_network

    model = neural_networks.get_network_model(network_name)

    print("Training new model: " + network_name)
    augmentation_mode = neural_networks.get_augmentation_mode(network_name)
    train_x, train_y = augment_data(augmentation_mode, train_x, train_y)

    model.fit(train_x, train_y, epochs=training_epochs, verbose=0)

    networks_repository.save_network(model, network_name)

    return model


def augment_data(image_generator, train_x, train_y):
    if image_generator is None:
        return train_x, train_y

    return image_generator.flow(train_x, train_y), None
