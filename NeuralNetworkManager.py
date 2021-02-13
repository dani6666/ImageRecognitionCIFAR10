from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD


class NeuralNetworkManager:
    networks_names = ["Simple CNN", "Flat CNN", "Simple CNN with more filters"]

    @staticmethod
    def get_network_model(name):
        if name == NeuralNetworkManager.networks_names[0]:
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        if name == NeuralNetworkManager.networks_names[1]:
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        if name == NeuralNetworkManager.networks_names[2]:
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
            return model
