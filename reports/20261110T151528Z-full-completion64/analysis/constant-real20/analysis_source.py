"""Verify the completed constant-LR control and create its local figures."""
import csv
import argparse
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
PROJECT = next(p for p in Path(__file__).resolve().parents
               if (p / 'model_collapse').is_dir() and (p / 'pyproject.toml').exists())
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_collapse.analysis.completion import _collect
from model_collapse.common import atomic_json, read_json, sha256
from model_collapse.cloud.execution import accrued, ledger

run = '20261110T151528Z-full-completion64'
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path)
args = parser.parse_args()
root = Path('runs') / run
child = root.with_name(run + '-real')
c = read_json(root / 'config.json')
assert read_json(root / 'completion.json')['complete']
assert c['generations'] == 20 and c['learning_rate'] == c['min_learning_rate'] == 5e-6
assert c['endpoint_arms'] == ['real']
rows, _, _ = _collect(root)
inputs = {}
for path in [root / name for name in ('config.json', 'resources.json', 'completion.json',
        'artifact_index.json', 'provider_shutdown.json', 'watchdog_exit.json', 'environment.json')]:
    inputs[path.as_posix()] = sha256(path)
for epoch in range(21):
    path = child / f'seed_42/gen_{epoch:02d}/metrics.json'
    inputs[path.as_posix()] = sha256(path)
assert len(rows) == 21 and [r['cycle'] for r in rows] == list(range(21))
historical_input_path = Path('runs/20261108T173505Z-full-completion64-real/seed_42/gen_01/training_input/manifest.json')
historical_input = read_json(historical_input_path)
inputs[str(historical_input_path)] = sha256(historical_input_path)
previous = rows[0]['weights_sha256']
assert previous == historical_input['lineage']['initializer_weights_sha256']
optimizer_parent = None
prompts, targets, token_inputs = set(), set(), set()
for epoch in range(1, 21):
    stage = child / f'seed_42/gen_{epoch:02d}'
    t = read_json(stage / 'training.json')
    h = read_json(stage / 'training_history.json')
    m = read_json(stage / 'training_input/manifest.json')
    for name in ('training.json', 'training_history.json', 'training_input/manifest.json'):
        inputs[(stage / name).as_posix()] = sha256(stage / name)
    assert t['initial_weights_sha256'] == previous
    assert t['selected_weights_sha256'] == t['terminal_weights_sha256'] != previous
    assert t['optimizer_parent_sha256'] == optimizer_parent
    assert t['global_step'] == epoch * 79 and t['total_optimizer_steps'] == 1580
    assert t['training_blocks'] == 10000 and t['executed_epochs'] == t['selected_epoch'] == 1
    assert len(h) == 1 and h[0]['scored_training_tokens'] == 640000
    assert h[0]['learning_rate'] == h[0]['learning_rate_end'] == 5e-6
    assert m['lineage']['source_collection'] == 'cycle_01'
    assert m['lineage']['training_epoch'] == epoch
    assert m['lineage']['generator_weights_sha256'] is None
    assert m['pairs'] == 10000 and m['scored_targets'] == 640000
    prompts.add(m['prompt_content_sha256']); targets.add(m['target_content_sha256'])
    token_inputs.add(t['input_tokens_sha256'])
    previous = t['selected_weights_sha256']
    optimizer_parent = t['state_files']['optimizer_state/cycle.pt']
assert len(prompts) == len(targets) == len(token_inputs) == 1
assert prompts == {historical_input['prompt_content_sha256']}
assert targets == {historical_input['target_content_sha256']}
for epoch, row in enumerate(rows):
    metrics = read_json(child / f'seed_42/gen_{epoch:02d}/metrics.json')
    for split, count in [('test', 457664), ('validation', 468608), ('fixed_real_anchor', 640000)]:
        v = metrics[split]
        assert v['scored_tokens'] == count
        assert math.isclose(v['loss'], v['nll_sum'] / count, rel_tol=1e-12)
        assert math.isclose(v['perplexity'], math.exp(v['loss']), rel_tol=1e-12)
restore = read_json(root / 'terminal_restores' / (child.name + '.json'))
inputs[(root / 'terminal_restores' / (child.name + '.json')).as_posix()] = sha256(root / 'terminal_restores' / (child.name + '.json'))
assert restore['scope'] == 'complete_terminal_stage_only'
assert restore['weights_sha256'] == previous and restore['test_loss_absolute_error'] <= 1e-6
guard = read_json(root / 'watchdog_exit.json')
assert guard['reason'] == 'complete'

out = args.out or Path('reports') / run / 'analysis/constant-real20'
assert not out.exists(), 'Preserve existing generated evidence; choose a new destination.'
out.mkdir(parents=True)
atomic_json(out / 'metrics.json', rows)
with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False})
x = [r['cycle'] for r in rows]
def axes(title, ylabel):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_box_aspect(1)
    ax.set(title=title, xlabel='Completed epochs on the same 10,000 real pairs', ylabel=ylabel)
    ax.set_xticks(range(0, 21, 2)); ax.grid(alpha=.2)
    return fig, ax
fig, ax = axes('Repeated real data: constant learning rate 5e-6', 'Corpus perplexity on fixed real continuations')
ax.plot(x, [r['test_perplexity'] for r in rows], 'o-', color='#176b87', label='Fixed test set', lw=2)
ax.plot(x, [r['validation_perplexity'] for r in rows], 's--', color='#a45898', label='Fixed validation set', lw=1.8, ms=4)
ax.legend(); fig.tight_layout(); fig.savefig(out / 'heldout_perplexity.png', dpi=180); plt.close(fig)
fig, ax = axes('Fitting repeated real text versus held-out text', 'Token-weighted negative log-likelihood (nats)')
for key, label, color in [('fixed_real_anchor_loss','Repeated training set','#176b87'),
                          ('test_loss','Fixed test set','#d0792d'), ('validation_loss','Fixed validation set','#a45898')]:
    ax.plot(x, [r[key] for r in rows], 'o-', color=color, label=label, ms=4)
ax.legend(); fig.tight_layout(); fig.savefig(out / 'training_vs_heldout_nll.png', dpi=180); plt.close(fig)
old = Path('reports/20261108T173505Z-full-completion64/analysis/four-condition/metrics.csv')
with old.open(encoding='utf-8') as f:
    historical = [r for r in csv.DictReader(f) if r['condition'] == 'repeated_real']
assert len(historical) == 11
fig, ax = axes('Repeated real-data controls', 'Fixed-test corpus perplexity')
ax.plot(x, [r['test_perplexity'] for r in rows], 'o-', color='#176b87', label='Constant 5e-6: 20 epochs')
ax.plot([int(r['cycle']) for r in historical], [float(r['test_perplexity']) for r in historical],
        's--', color='#8b6a46', label='Historical cosine 5e-6 to 1e-6: 10 epochs', ms=4)
ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out / 'constant_vs_historical_cosine.png', dpi=180); plt.close(fig)
inputs[str(old)] = sha256(old)
shutil.copyfile(__file__, out / 'analysis_source.py')
d = ledger()
launches = [a for a in d['launches'] if a.get('allocation') == 'fineweb_completion64_repeated_chunk_real_20epochs']
ids = {a['volume_id'] for a in launches}
attempts = {p.parent.name for p in Path('runs').glob('20261110*/config.json')
            if read_json(p).get('experiment_id') == c['experiment_id']}
volumes = {k:v for k,v in d.get('volumes',{}).items() if k in ids or
           v.get('allocation') == 'fineweb_completion64_repeated_chunk_real_20epochs' or
           v.get('run_id') in attempts}
cost = accrued({**d, 'launches': launches, 'volumes': volumes})
best = min(rows, key=lambda r:r['test_perplexity'])
summary = {'run_id':run,'epochs':20,'optimizer_updates':1580,'target_token_exposures':12800000,
    'baseline_test_perplexity':rows[0]['test_perplexity'],'minimum_test_perplexity':best['test_perplexity'],
    'minimum_at_epoch':best['cycle'],'final_test_perplexity':rows[-1]['test_perplexity'],
    'baseline_training_nll':rows[0]['fixed_real_anchor_loss'],'final_training_nll':rows[-1]['fixed_real_anchor_loss'],
    'constant_lr':5e-6,'all_initializer_optimizer_and_repeated_input_checks':True,
    'validation_and_test_counts_frozen':True,'provider_shutdown':read_json(root/'provider_shutdown.json'),
    'watchdog_reason':guard['reason'],'terminal_restore':restore,
    'task_cost_conservative_at_receipt_usd':cost,'cost_observed_at':time.time(),
    'retained_working_plus_backup_storage_usd_per_day':60*(.20+.07)/730*24}
atomic_json(out/'summary.json',summary)
text = '# Constant learning rate on repeated real data\n\n'
text += ('Original pretrained OPT-125M, twenty continued epochs on the identical first 10,000 FineWeb pairs. '
         'AdamW, optimizer/RNG carry, continuation-only 64-token loss and fixed evaluation are preserved. '
         'Learning rate is 5e-6 throughout all 1,580 updates. There is no reset between epochs.\n\n')
text += f"Test perplexity: {rows[0]['test_perplexity']:.5f} initially, minimum {best['test_perplexity']:.5f} at epoch {best['cycle']}, final {rows[-1]['test_perplexity']:.5f}.\n\n"
text += f"Fixed training NLL: {rows[0]['fixed_real_anchor_loss']:.5f} to {rows[-1]['fixed_real_anchor_loss']:.5f}.\n\n"
text += ('Falling training loss with rising held-out loss is consistent with overfitting. '
         'This is a single-seed finite trajectory; perplexity alone does not uniquely diagnose synthetic-data collapse. '
         'The historical comparison has ten epochs, a decaying learning rate and RTX PRO 4000 hardware; '
         'the new run has twenty epochs and RTX 2000 Ada hardware. These are recorded differences.\n\n')
text += ('All twenty checkpoints and optimization states remain in private backups. The terminal stage was cold-restored and '
         're-evaluated on the GPU; this does not claim cold restoration of all twenty bulk stages. '
         'GPU compute is stopped; the working disk and backup volume remain retained.\n\n')
text += f'Conservative task cost at this receipt: USD {cost:.4f}; retained new storage approximately USD {summary["retained_working_plus_backup_storage_usd_per_day"]:.2f}/day.\n'
text += ('\nReproduce from the recovered raw run with `python ' +
         str(out / 'analysis_source.py').replace('\\', '/') +
         ' --out reports/' + run + '/analysis/reproduction-NEW`. Existing evidence is never overwritten.\n')
(out/'README.md').write_text(text,encoding='utf-8')
atomic_json(out/'analysis_manifest.json', {'created_at':time.time(),'inputs':inputs,
    'input_paths_relative_to':'project root',
    'analysis_source_sha256':sha256(__file__),'python':platform.python_version(),'matplotlib':matplotlib.__version__,
    'training_runtime_sha256':read_json(root/'resources.json')['runtime_sha256'],
    'outputs':{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})
print({k:v for k,v in summary.items() if k not in ('terminal_restore','provider_shutdown')})
