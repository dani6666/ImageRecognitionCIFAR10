from NeuralNetworkComparer import NeuralNetworkComparer
from NeuralNetworkManager import NeuralNetworkManager


def main():
    NeuralNetworkComparer.compare_neural_networks(NeuralNetworkManager.get_all_networks())


if __name__ == "__main__":
    main()
