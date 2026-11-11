"""Offline cycle-10 metrics against actual synthetic-target percentages."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.analysis.publication import new_destination, verified_publication, write_provenance


FRACTIONS = (0., .5, .8, .95, .99, .999, 1.)
COLORS = ('#0072B2', '#E69F00', '#D55E00', '#B32043', '#222222', '#007F86', '#CC79A7')
MARKERS = ('o', '^', 'D', 'X', '*', 'H', 'o')
PANELS = (
    ('test_perplexity', 'Final real-test perplexity', 'Corpus perplexity (lower is better)', 1., '01_final_perplexity'),
    ('token_entropy_nats', 'Final pooled token entropy', 'Empirical pooled entropy (nats)', 1., '02_final_pooled_entropy'),
    ('mean_within_entropy_nats', 'Final entropy within outputs', 'Mean per-output entropy (nats)', 1., '03_final_within_entropy'),
    ('distinct_2', 'Final distinct-2', 'Unique bigrams / all bigrams (%)', 100., '04_final_distinct2'),
    ('distinct_3', 'Final distinct-3', 'Unique trigrams / all trigrams (%)', 100., '05_final_distinct3'),
    ('mean_within_repeated_4gram_rate', 'Final repeated four-token phrases', 'Mean within-output repeated 4-grams (%)', 100., '06_final_repetition'),
)


def build(source, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    source = Path(source).resolve()
    verified_publication(source)
    metrics = source / 'analysis/seven-fractions/metrics.json'
    records = json.loads(metrics.read_text(encoding='utf-8'))
    selected = sorted((r for r in records if r['cycle'] == 10), key=lambda r: r['synthetic_fraction'])
    if tuple(r['synthetic_fraction'] for r in selected) != FRACTIONS:
        raise ValueError('Require exactly the seven declared cycle-10 endpoints')
    if any(r['global_step'] != 790 or r['selected_epoch'] != 1 for r in selected):
        raise ValueError('All endpoints must follow ten terminal epochs and 790 updates')
    destination = new_destination(source, out)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'cycle10_metrics.json').write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
    with (destination / 'cycle10_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.labelsize': 13,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18})
    x = np.asarray([100 * r['synthetic_fraction'] for r in selected])
    for key, title, ylabel, multiplier, name in PANELS:
        y = np.asarray([multiplier * r[key] for r in selected])
        fig, ax = plt.subplots(figsize=(8.6, 8.6))
        fig.subplots_adjust(left=.15, right=.96, top=.78, bottom=.18)
        fig.suptitle(title + ' versus synthetic share', y=.965, fontsize=17, fontweight='bold')
        ax.plot(x, y, color='#8b8b8b', ls='--', lw=1.3, zorder=1)
        for i, r in enumerate(selected):
            ax.scatter(x[i], y[i], s=80 if i == 4 else 62, marker=MARKERS[i],
                       color=COLORS[i], edgecolors='white', linewidths=.6, zorder=3,
                       label=f'{x[i]:.12g}% synthetic')
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel('Synthetic continuation targets (%)')
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, .915), ncol=2,
                   fontsize=10.5, frameon=False)
        if key == 'test_perplexity':
            ax.set_ylim(15, 365)
            for i, offset in enumerate(((8, 6), (-7, 8), (-12, 8), (-13, 7), (-15, 4), (-15, 4), (-15, -13))):
                ax.annotate(f'{y[i]:.2f}', (x[i], y[i]), xytext=offset, textcoords='offset points',
                            ha='right' if i else 'left', fontsize=10, color=COLORS[i])
            inset = ax.inset_axes([.10, .40, .57, .48])
            indices = list(range(4))
            inset.set_xlim(-3, 98)
            inset.set_ylim(27, 40)
            inset.set_xticks([0, 50, 80, 95])
            inset.set_title('Closer view: 0–95%', fontsize=10)
        else:
            upper = key == 'mean_within_repeated_4gram_rate'
            pooled = key == 'token_entropy_nats'
            inset = ax.inset_axes([.10 if not pooled else .50, .53 if upper else .12, .46, .34])
            indices = list(range(3, 7))
            inset.set_xlim(94.5, 100.4)
            inset.set_xticks([95, 97, 99, 100])
            inset.set_title('Closer view: 95–100%', fontsize=10)
            subset_y = y[indices]
            span = max(float(subset_y.max() - subset_y.min()), .001)
            inset.set_ylim(float(subset_y.min()) - .12*span, float(subset_y.max()) + .12*span)
        inset.plot(x[indices], y[indices], color='#8b8b8b', ls='--', lw=1, zorder=1)
        for i in indices:
            inset.scatter(x[i], y[i], color=COLORS[i], marker=MARKERS[i], s=44, zorder=3)
        inset.tick_params(labelsize=9)
        inset.set_xlabel('Synthetic targets (%)', fontsize=9)
        inset.set_facecolor('white')
        fig.text(.5, .035,
            'All values are from cycle 10; both axes use actual numerical values.\n'
            'Dashed joins are visual guides, not fits or evidence about untested fractions.\n'
            'Seed 42 · shared cosine LR 5e-6 → 1e-6 · 6.4 million targets per arm\n'
            '95/99/99.9%: RTX PRO 4500; other shown conditions: RTX PRO 4000',
            ha='center', fontsize=8.7, color='#555555', linespacing=1.6)
        for extension in ('png', 'pdf'):
            fig.savefig(destination / f'{name}.{extension}', dpi=200, facecolor='white')
        plt.close(fig)
    write_provenance(source, destination, __file__, [metrics, source / 'publication_manifest.json'],
        scope='Seven fixed cycle-10 endpoints; actual numeric fraction spacing, no fits and no new model evaluation.')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.out))
