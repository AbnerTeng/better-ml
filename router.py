from argparse import ArgumentParser, Namespace

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from sklearn_best_practice.gb_trainer import RFTrainer
from dt_from_scratch.id3 import ID3
from src.datasets import load_toy
from src.utils import load_config
from pytorch_tutorial.src.expr import reg_expr

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--expr",
        type=str,
        required=True
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.expr == "scratch_id3":
        toy = load_toy()
        X, y = toy.data, toy.target
        x_tr, x_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        id3_algo = ID3(max_depth=3)
        id3_algo.fit(x_tr, y_tr)
        y_pred = id3_algo.predict(x_te)
        print(f"Accuracy: {accuracy_score(y_te, y_pred)}")

    elif args.expr == "sklearn_best_practice":
        rf_config = load_config("config/rf_config.yml")
        iris = load_iris()
        X, y = iris.data, iris.target
        x_tr, x_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        rf = RFTrainer(RandomForestClassifier, rf_config)
        y_pred = rf.train(x_tr, y_tr, x_te)
        rf.evaluate(y_pred, y_te, issue="clf")

    elif args.expr == "reg_nn":
        reg_expr.main()

    elif args.expr == "rnn":
        pass
    else:
        raise ValueError(f"Invalid expr: {args.expr}")
