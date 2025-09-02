import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable


def poly(x):
    return 6.65 * x - 29.5 * x ** 2 + 46.0 * x ** 3 - 22.41 * x ** 4


def get_dataset(n=250, seed=0):
    X = np.random.default_rng(seed).random(size=(n, 2))
    y = poly(X[:, 0]) <= X[:, 1]
    return X, y


def subplots(**arg):
    fig, ax = plt.subplots(**arg)
    if isinstance(ax, Iterable):
        for axx in ax:
            setup_axis(axx)
    else:
        setup_axis(ax)
    return fig, ax


def setup_axis(ax):
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


default_style = [
    {
        's': 600,
        'marker': 'o',
        'c': 'red',
        'alpha': 0.9,
        'edgecolor': 'black',
    },
    {
        's': 600,
        'marker': 'o',
        'c': 'blue',
        'alpha': 0.9,
        'edgecolor': 'black',
    },
]


def scatter_pts(ax, X, y, style=None, shuffle_points=True):
    if style is None:
        style = default_style

    if shuffle_points:
        # Plot each point one by one (slower, but creates a more natural plot in areas where points overlap)
        order = np.random.default_rng(42).permutation(range(len(X)))
        for i in order:
            cls = y[i]
            ax.scatter(X[i, 0], X[i, 1], s=style[cls]['s'], marker=style[cls]['marker'],
                       c=style[cls]['c'], alpha=style[cls]['alpha'], edgecolor=style[cls]['edgecolor'])
    else:
        # Plot all of class 0, then all of class 1. If there's any overlap, class 1 will always be on-top
        for cls in np.unique(y):
            ax.scatter(X[y == cls][:, 0], X[y == cls][:, 1], s=style[cls]['s'], marker=style[cls]['marker'],
                       c=style[cls]['c'], alpha=style[cls]['alpha'], edgecolor=style[cls]['edgecolor'])


def contour(axis, predictor, **kwargs):
    res = 201
    xx, yy, grid = make_grid()
    axis.contour(xx, yy, predictor.predict(grid).reshape((res, res)), levels=[0.5], colors=['black'], linewidths=8, **kwargs)


def make_grid(x_min=0, x_max=1, y_min=0, y_max=1, res=201):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    return xx, yy, np.array((xx.ravel(), yy.ravel())).T
