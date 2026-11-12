"""Render connected endpoint views from immutable saved metrics, offline."""
import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_collapse.common import atomic_json, read_json, sha256
from scripts.analysis.build_smollm_fraction_report import ARMS, COLORS, LABELS
from scripts.analysis.publication import new_destination

METRICS = [
    (13, 'test_perplexity', 'Cycle-10 perplexity versus synthetic fraction', 'Corpus continuation perplexity', False),
    (14, 'token_entropy_nats', 'Cycle-10 pooled entropy versus synthetic fraction', 'Pooled token entropy (nats)', False),
    (15, 'mean_within_entropy_nats', 'Cycle-10 local entropy versus synthetic fraction', 'Mean within-output entropy (nats)', False),
    (16, 'distinct_2', 'Cycle-10 distinct-2 versus synthetic fraction', 'Pooled distinct-2 (%)', True),
    (17, 'distinct_3', 'Cycle-10 distinct-3 versus synthetic fraction', 'Pooled distinct-3 (%)', True),
    (18, 'mean_within_repeated_4gram_rate', 'Cycle-10 repetition versus synthetic fraction', 'Mean within-output repeated 4-grams (%)', True),
]


def render(metrics, out):
    source = Path(metrics).resolve()
    original_hash = sha256(source)
    rows = read_json(source)
    final = {row['arm']: row for row in rows if row['cycle'] == 10}
    assert set(final) == set(ARMS)
    x = [100 * final[arm]['synthetic_fraction'] for arm in ARMS]
    assert x == sorted(x) and len(set(x)) == 7
    destination = new_destination(source.parent, out)
    destination.mkdir(parents=True)
    plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white',
                         'font.family': 'DejaVu Sans'})
    figures = []
    for number, field, title, ylabel, percent in METRICS:
        y = [final[arm][field] * (100 if percent else 1) for arm in ARMS]
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.set_box_aspect(1)
        ax.plot(x, y, color='#7B858E', linewidth=1.6, zorder=1)
        for i, arm in enumerate(ARMS):
            ax.scatter(x[i], y[i], s=65, color=COLORS[arm], zorder=3,
                       label=LABELS[arm] + ' synthetic')
        ax.set(title=title, xlabel='Synthetic training continuations (%)',
               ylabel=ylabel, xlim=(-3, 104))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        fig.subplots_adjust(left=.15, right=.96, bottom=.14, top=.90)
        ax.title.set_fontsize(11.5)
        ax.title.set_wrap(True)
        name = f'{number:02d}_endpoint_{field}'
        fig.savefig(destination / (name + '.png'), dpi=180)
        fig.savefig(destination / (name + '.pdf'),
                    metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
        figures.append({'name': name, 'x': x, 'y': y,
                        'metric': field, 'percentage_display': percent})
    assert sha256(source) == original_hash
    (destination / 'README.md').write_text(
        '# Connected endpoint figures\n\n'
        'Presentation-only revision of the six SmolLM2 cycle-10 endpoint figures. '
        'Straight segments connect the original measured points in increasing '
        'synthetic-fraction order. Values, axes, dot colours and square sizes are '
        'unchanged; the original publication remains preserved. The three endpoint '
        'figures used in Google Docs are replaced in place.\n', encoding='utf-8')
    atomic_json(destination / 'figure_revision.json', {
        'source_metrics_sha256': original_hash, 'script_sha256': sha256(__file__),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'data_unchanged': True, 'figures': figures,
        'outputs': {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}})
    print('Rendered six connected endpoint figures; source metrics unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--out', required=True)
    render(**vars(parser.parse_args()))
