from argparse import ArgumentParser, Namespace
import numpy as np
import yfinance as yf


def get_args() -> Namespace:
    """
    Get arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--ticker",
        "-t",
        type=str,
        default="0050.TW"
    )
    parser.add_argument(
        "--start",
        "-st",
        type=str,
        default="2020-01-01"
    )
    parser.add_argument(
        "--end",
        "-e",
        type=str,
        default="2024-01-01"
    )
    return parser.parse_args()


if __name__  == "__main__":
    args = get_args()
    data = yf.download(args.ticker, start=args.start, end=args.end)
    data.drop(columns=["Adj Close"], inplace=True)
    mat = data.to_numpy()
    np.save("data/stock.npy", mat)
