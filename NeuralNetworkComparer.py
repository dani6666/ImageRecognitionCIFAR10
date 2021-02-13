from keras.datasets import cifar10
from keras.utils import to_categorical


class NeuralNetworkComparer:

    @staticmethod
    def compare_neural_networks(neural_networks):
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()

        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        train_x = train_x / 255.0
        test_x = test_x / 255.0

        for neural_network in neural_networks:
            print("Testing " + neural_network.name)

            history = neural_network.model.fit(train_x, train_y, epochs=20, batch_size=64, validation_data=(train_x, train_y), verbose=0)

            _, accuracy = neural_network.model.evaluate(test_x, test_y, verbose=0)

            print("Accuracy: " + str(accuracy))

