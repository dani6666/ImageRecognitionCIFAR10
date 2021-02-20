import neural_networks
import data
import repository

training_epochs = 50


def retrain_networks(network_names):

    for network_name in network_names:
        repository.delete_network(network_name)

    train_x, train_y = data.get_cifar10_train_data()

    for network_name in network_names:
        model = neural_networks.get_network_model(network_name)

        print("Retraining network: " + network_name)
        augmentation_mode = neural_networks.get_augmentation_mode(network_name)
        train_x, train_y = augment_data(augmentation_mode, train_x, train_y)

        model.fit(train_x, train_y, epochs=training_epochs, verbose=2)

        repository.save_network(model, network_name)


def train_all_networks(network_names):

    train_x, train_y = data.get_cifar10_train_data()

    for network_name in network_names:
        if not repository.network_exists(network_name):
            model = neural_networks.get_network_model(network_name)

            print("Training new network: " + network_name)
            augmentation_mode = neural_networks.get_augmentation_mode(network_name)
            train_x, train_y = augment_data(augmentation_mode, train_x, train_y)

            model.fit(train_x, train_y, epochs=training_epochs, verbose=2)

            repository.save_network(model, network_name)


def get_trained_network(network_name, train_x, train_y):

    saved_network = repository.get_network(network_name)
    if saved_network is not None:
        return saved_network

    model = neural_networks.get_network_model(network_name)

    print("Training new model: " + network_name)
    augmentation_mode = neural_networks.get_augmentation_mode(network_name)
    train_x, train_y = augment_data(augmentation_mode, train_x, train_y)

    model.fit(train_x, train_y, epochs=training_epochs, verbose=0)

    repository.save_network(model, network_name)

    return model


def augment_data(image_generator, train_x, train_y):
    if image_generator is None:
        return train_x, train_y

    return image_generator.flow(train_x, train_y), None
