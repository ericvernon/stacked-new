import os
from pathlib import Path

import pandas as pd


def load_dataset(dataset_id):
    # Load a file, assuming the first (N - 1) columns are features, and the last column is the target value
    my_dir = os.path.dirname(os.path.realpath(__file__))
    path = Path(f'../data/uci/{dataset_id}')
    df = pd.read_csv(my_dir / path / 'data.txt')

    arr = df.to_numpy()
    X, y = arr[:, :-1], arr[:, -1]
    return X, y


if __name__ == '__main__':
    load_dataset(94)
