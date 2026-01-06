import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adjustText import adjust_text
from pathlib import Path

reports_root = Path('./output/reports')
folders = [
    'CW_1x5_Dynamic_Complete/20251212-182251',
    'CW_1x5_Static_Midscope/20251218-174146',
    'CW_3x5_Dynamic_Midscope/20251216-190628',
    'CW_3x5_Static_Midscope/20251218-162436',
    'CW_5x5_Dynamic_Midscope/20251216-190523',
    'CW_5x5_Static_Midscope/20251218-162132',
]
midscope_datasets = [
    17, 19, 43, 52, 94, 96, 151, 174, 176, 212, 267,
    329, 451, 519, 537, 545, 563, 850, 863, 887, 891
]
# All, Shallow, Deep
compare = 'Deep'
show_labels = [
    '1x5_Dynamic_Double_GB_Shallow',
    '1x5_Dynamic_Double_AR_Deep',
    '1x5_Static_Double_GB_Deep',
    '5x5_Dynamic_Double_AR_Shallow',
    '1x5_Dynamic_Ternary_Shallow',
    '1x5_Dynamic_Ternary_Deep',
    '1x5_Dynamic_Double_GB_Deep',
    '3x5_Static_Double_GB_Deep',
]

def main():
    dfs = []
    for folder in folders:
        path = reports_root / folder / 'summary.txt'
        df = pd.read_csv(path)
        # drop any binary graders from this comparison
        df = df[~df['algorithm'].str.contains('Binary')]
        # drop any extra datasets
        df = df[df['dataset'].isin(midscope_datasets)]
        # hacky filtering -- to show pareto front for only shallow vs. deep grader types
        if compare == 'Shallow':
            df = df[~df['algorithm'].str.contains('Deep')]
        elif compare == 'Deep':
            df = df[~df['algorithm'].str.contains('Shallow')]
        dfs.append(df)
    df = pd.concat(dfs)
    ranks_train = ranks_table(df, 'train')
    ranks_test = ranks_table(df, 'test')

    pf_train = plot_pareto_front(ranks_train, 'train')
    pf_test = plot_pareto_front(ranks_test, 'test')

    print("TRAINING TABLE:")
    print(table_to_latex(ranks_train, pf_train))
    print("")
    print("TESTING TABLE:")
    print(table_to_latex(ranks_test, pf_test))


def ranks_table(df, train_or_test):
    rank_specs = {
        f'hybrid_accuracy_{train_or_test}': 'max',
        f'hybrid_reject_{train_or_test}': 'min',
        f'hybrid_kappa_{train_or_test}': 'max',
    }

    for metric, best in rank_specs.items():
        ascending = (best == 'min')  # True if lower=better
        df[f'{metric}_rank'] = df.groupby('dataset')[metric].rank(
            ascending=ascending, method='average'
        )

    summary = (
        df.groupby('algorithm')
        .agg(
            mean_accuracy=(f'hybrid_accuracy_{train_or_test}', 'mean'),
            mean_reject=(f'hybrid_reject_{train_or_test}', 'mean'),
            mean_kappa=(f'hybrid_kappa_{train_or_test}', 'mean'),
            mean_accuracy_rank=(f'hybrid_accuracy_{train_or_test}_rank', 'mean'),
            mean_reject_rank=(f'hybrid_reject_{train_or_test}_rank', 'mean'),
            mean_kappa_rank=(f'hybrid_kappa_{train_or_test}_rank', 'mean'),
        )
        .reset_index()
    )
    return summary.sort_values(by='mean_accuracy', ascending=False)


def table_to_latex(summary_table, pf):
    sb = ''
    for idx, row in summary_table.iterrows():
        # bold = row["algorithm"] in pf.values
        bold = row["algorithm"] == "1x5_Dynamic_Double_GB_Shallow"
        algo_name = bold_helper(row["algorithm"].replace("_", "-"), bold)
        mean_accuracy = bold_helper(f'{(100 * row["mean_accuracy"]):.2f}', bold)
        mean_reject = bold_helper(f'{(100 * row["mean_reject"]):.2f}', bold)
        mean_kappa = bold_helper(f'{row["mean_kappa"]:.2f}', bold)
        mean_acc_rank = bold_helper(f'{row["mean_accuracy_rank"]:.1f}', bold)
        mean_reject_rank = bold_helper(f'{row["mean_reject_rank"]:.1f}', bold)
        mean_kappa_rank = bold_helper(f'{row["mean_kappa_rank"]:.1f}', bold)

        sb += (
            f'{algo_name} & '
            f'{mean_accuracy}\\% & '
            f'{mean_reject}\\% & '
            f'{mean_kappa} & '
            f'{mean_acc_rank} & '
            f'{mean_reject_rank} & '
            f'{mean_kappa_rank} \\\\ \\hline \n'
        )
    return sb


def bold_helper(text, bold):
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def plot_pareto_front(summary_df, train_or_test):
    x_col, max_x = 'mean_reject', False
    y_col, max_y = 'mean_accuracy', True
    frontier = pareto_frontier(summary_df, x_col, y_col, max_x, max_y)
    plt.figure(figsize=(8, 8))
    plt.scatter(summary_df[x_col], summary_df[y_col], c='gray', alpha=0.6)
    plt.plot(frontier[x_col], frontier[y_col], 'r-o', linewidth=1)

    # Label points
    texts = []
    for _, row in summary_df.iterrows():
        if row['algorithm'] in show_labels or compare != 'All':
            texts.append(
                plt.text(row[x_col], row[y_col], f'{row['algorithm']}', fontsize=12)
            )
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if train_or_test == 'train':
        words = 'Training Set'
    else:
        words = 'Testing Set'
    if compare == 'Shallow':
        words += ', Shallow Graders Only'
    elif compare == 'Deep':
        words += ', Deep Graders Only'
    plt.title(f'Accuracy vs. Reject Rate ({words})', fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'figv2/out/grader_comparison_{train_or_test}_compare_{compare}.png',
        dpi=300,  # High resolution (dots per inch)
        bbox_inches='tight',  # Trims whitespace and prevents cutting off labels
        transparent=True  # Transparent background (PNG only)
    )
    plt.show()
    return frontier['algorithm']


def pareto_frontier(df, x_col, y_col, maximize_x=False, maximize_y=True):
    df_sorted = df.sort_values(by=x_col, ascending=not maximize_x)
    pareto = []
    best_y = -np.inf if maximize_y else np.inf

    for _, row in df_sorted.iterrows():
        y = row[y_col]
        if maximize_y and y > best_y or not maximize_y and y < best_y:
            pareto.append(row)
            best_y = y
    return pd.DataFrame(pareto)


if __name__ == '__main__':
    main()
