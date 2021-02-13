from NeuralNetworkComparer import NeuralNetworkComparer
from AllNeuralNetworks import AllNeuralNetworks
from NeuralNetworksManager import NeuralNetworksManager
import argparse


def main(args):
    if args.train:
        NeuralNetworksManager.train_all_networks(AllNeuralNetworks.networks_names)
    else:
        NeuralNetworkComparer.compare_neural_networks(AllNeuralNetworks.networks_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='train', action='store_true',
                        help='Train all untrained models without comparing')

    main(parser.parse_args())
