from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf

from ..dataset.ts_dataset import build_dataloader
from ..model.rnn import RNN
from ..train import SequenceTrainer


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--conf_path", "-cp", type=str, default="config/rnn_config.yaml"
    )

    return parser.parse_args()


def main():
    args = get_args()
    conf = OmegaConf.load(args.conf_path)
    train_dataloader, input_dim, output_dim = build_dataloader(
        tags="train", data_conf=conf.data
    )
    val_dataloader, _, _ = build_dataloader(tags="val", data_conf=conf.data)
    test_dataloader, _, _ = build_dataloader(tags="test", data_conf=conf.data)

    model = RNN(input_dim, conf.model.hidden_dim, output_dim, conf.model.num_layers)
    trainer = SequenceTrainer(model, conf.train)
    trainer.train(train_dataloader, val_dataloader)
    trainer.predict(test_dataloader, trainer.model)


if __name__ == "__main__":
    main()
