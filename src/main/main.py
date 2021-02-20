from src.repository import neural_networks, networks_repository
from src.core import trainer, comparer
import argparse


def main(args):

    if args.delete:
        networks_repository.delete_all_networks()
    elif any(args.retrain_networks):
        trainer.retrain_networks(args.retrain_networks)
    elif args.train:
        trainer.train_all_networks(neural_networks.networks_names)
    else:
        comparer.compare_neural_networks(neural_networks.networks_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='train', action='store_true', help='Train all untrained models without comparing')
    parser.add_argument('-d', dest='delete', action='store_true', help='Delete all trained models')
    parser.add_argument('-r', dest='retrain_networks', nargs="+", default=[], help='Choose network names to retrain')

    parsed_args = parser.parse_args()

    used_args = 0

    if parsed_args.train:
        used_args += 1
    if parsed_args.delete:
        used_args += 1
    if any(parsed_args.retrain_networks):
        used_args += 1

    if used_args > 1:
        print("You can use only one of additional args at one run")
        exit(1)

    main(parsed_args)
