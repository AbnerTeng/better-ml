from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf

from ..dataset.tab_reg_dataset import build_dataloader
from ..model.vanilla_nn import VanillaNN
from ..train import Trainer


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--conf_path",
        '-cp',
        type=str,
        default="config/reg_config.yaml"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    conf = OmegaConf.load(args.conf_path)
    train_dataloader, input_dim, output_dim = build_dataloader(tags="train", data_conf=conf.data)
    val_dataloader = build_dataloader(tags="val", data_conf=conf.data)
    test_dataloader = build_dataloader(tags="test", data_conf=conf.data)

    model = VanillaNN(
        input_dim,
        conf.model.hidden_dim,
        output_dim
    )
    trainer = Trainer(model, conf.train)
    trainer.train(train_dataloader, val_dataloader)
    trainer.predict(test_dataloader, trainer.model)
