import numpy as np
import pandas as pd

np.random.seed(0)


def make_dummy_data(n_classes=2, n_rows=15, glass_acc=0.7, black_acc=0.9, grader_weights=None, fn='./test.txt'):
    y_truth = np.random.choice(a=np.arange(n_classes), size=(n_rows,))

    y_glass = y_truth.copy()
    flip_idx = np.random.random(size=(n_rows,)) > (n_classes * glass_acc - 1) / (n_classes - 1)
    y_glass[flip_idx] = np.random.choice(a=np.arange(n_classes), size=(n_rows,))[flip_idx]

    y_black = y_truth.copy()
    flip_idx = np.random.random(size=(n_rows,)) > (n_classes * black_acc - 1) / (n_classes - 1)
    y_black[flip_idx] = np.random.choice(a=np.arange(n_classes), size=(n_rows,))[flip_idx]

    if grader_weights is None:
        grader_weights = [0.7, 0.3, 0.0]
    y_grader = np.random.choice(a=[0, 1, 2], size=(n_rows,), p=grader_weights)

    df = pd.DataFrame(data={
        'y_truth': y_truth,
        'y_glass': y_glass,
        'y_black': y_black,
        'y_grader': y_grader,
    })
    df.to_csv(fn, index=False)
    return df


if __name__ == '__main__':
    make_dummy_data(n_rows=20, n_classes=4, fn='./simple_binary.txt')
    make_dummy_data(n_rows=100_000, n_classes=4, fn='./simple_binary_long.txt')
    make_dummy_data(n_rows=20, n_classes=4, fn='./simple_ternary.txt', grader_weights=[0.5, 0.25, 0.25])
    make_dummy_data(n_rows=100_000, n_classes=4, fn='./simple_ternary_long.txt', grader_weights=[0.5, 0.25, 0.25])
