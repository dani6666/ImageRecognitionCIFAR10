from src.core import trainer, comparer

from src.main import argument_parser


def main():
    args = argument_parser.parse_args()

    if args.delete:
        trainer.delete_all_trained_networks()
    elif any(args.retrain_networks):
        trainer.retrain_networks(args.retrain_networks)
    elif args.train:
        trainer.train_all_networks()
    elif args.compare_only:
        comparer.compare_trained_neural_networks()
    else:
        comparer.compare_all_neural_networks()


if __name__ == "__main__":
    main()
