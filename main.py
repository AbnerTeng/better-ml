from typing import Any, Dict
import json

from argparse import ArgumentParser, Namespace
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from dt_from_scratch.id3 import ID3
from sklearn_best_practice.gb_trainer import RFTrainer
from datasets import load_toy


def load_config(path: str) -> Dict[str, Any]:
    """
    load model config
    """
    file_type = path.split(".")[-1]

    if file_type == "json":
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return config

    elif file_type in ["yaml", "yml"]:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config


def parsing_args() -> Namespace:
    """
    Parse the command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--lab",
        type=str,
        default="scratch_id3",
        help="The experiment to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parsing_args()

    if args.lab == "scratch_id3":
        toy = load_toy()
        X, y = toy.data, toy.target
        x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        id3_algo = ID3(max_depth=3)
        id3_algo.fit(x_tr, y_tr)
        y_pred = id3_algo.predict(x_te)
        print(f"Accuracy: {accuracy_score(y_te, y_pred)}")

    elif args.lab == "sklearn_module":
        rf_config = load_config("config/rf_config.yml")
        iris = load_iris()
        X, y = iris.data, iris.target
        x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RFTrainer(RandomForestClassifier, rf_config)
        y_pred = rf.train(x_tr, y_tr, x_te)
        rf.evaluate(y_pred, y_te, issue="clf")
