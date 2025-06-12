from pathlib import Path

import pandas as pd

def load_dataset(dataset_id):
    # Load a file, assuming the first (N - 1) columns are features, and the last column is the target value
    path = Path(f'../data/uci/{dataset_id}')
    df = pd.read_csv(path / 'data.txt')
    arr = df.to_numpy()
    X, y = arr[:, :-1], arr[:, -1]
    return X, y


if __name__ == '__main__':
    load_dataset(94)
