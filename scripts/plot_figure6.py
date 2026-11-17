"""Redraw verified Main Figure 5 (formerly R2 Figure 6) diagonal metrics separately from frozen paper artwork."""
from pathlib import Path
import csv
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (ROOT / 'results/additional/diagonal_comparison.csv').open(newline='') as handle:
        row = next(row for row in csv.DictReader(handle) if row['environment'] == 'uniform_partner')
    keys = ['tuna_corrected', 'symmetric_frozen_esm2_pair_mlp', 'pooled_ESM_logistic_sum_absdiff',
            'dscript_esm2', 'additive', 'rapppid']
    labels = ['TUnA', 'Frozen-ESM2', 'Logistic(sum+absdiff)', 'D-SCRIPT', 'Additive(sum only)', 'RAPPPID']
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(labels[::-1], [float(row[key + '_auroc']) for key in keys][::-1], color='#456889')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax.set(xlabel='Uniform / Uniform AUROC', xlim=(0, 0.8))
    fig.tight_layout()
    target = ROOT / 'results/figures_regenerated'
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / 'figure6_metrics_redraw.pdf', metadata={'Creator': '', 'Producer': '', 'CreationDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
