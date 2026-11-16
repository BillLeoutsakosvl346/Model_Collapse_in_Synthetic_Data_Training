"""Whole-epoch source removal at original batch weighting; no held-out gradients."""
from __future__ import annotations

import json
import math
import uuid
import time
from pathlib import Path

import numpy as np
import torch

from ..common import atomic_json, digest, read_json, sha256, verify_files
from .sgd_effects import preserve_rng


def validate_training_source(c, blocks, labels, source, order, observer):
    if (source not in ('both', 'real', 'synthetic') or c.get('optimizer') != 'sgd'
            or c.get('precision') != 'fp32' or c.get('epochs') != 1
            or c.get('learning_rate') != .01 or c.get('weight_decay') != 0
            or c.get('momentum') != 0 or c.get('update_diagnostics', {}).get('enabled')):
        raise ValueError('Source ablation requires its plain-SGD terminal-epoch policy')
    if labels is None or np.asarray(labels).shape != (len(blocks),) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Ablation source labels do not match training rows')
    if order is None or not np.array_equal(np.sort(np.asarray(order)), np.arange(len(blocks))):
        raise ValueError('Ablation row order must be a complete permutation')
    if c.get('epoch_ablation', {}).get('source') != source or observer is None:
        raise ValueError('Ablation contract/observer missing')


class SourceStep:
    """Same microbatch autograd arithmetic as historical SourceAccumulator.total."""
    def __init__(self, model, denominator, source):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.denominator, self.source = denominator, source
        self.counts = [0, 0]
        self.losses = [0., 0.]
        self.pairs = [0, 0]

    def backward(self, sums, counts, labels):
        labels = torch.as_tensor(labels, device=sums.device)
        counts = torch.as_tensor(counts, device=sums.device)
        selected = torch.ones_like(labels, dtype=torch.bool) if self.source == 'both' else labels == (self.source == 'synthetic')
        objective = sums.sum() if self.source == 'both' else sums[selected].sum()
        gradient = torch.autograd.grad(objective / self.denominator,
                                       self.parameters, allow_unused=True)
        for p, value in zip(self.parameters, gradient):
            if value is not None:
                if not torch.isfinite(value).all():
                    raise RuntimeError('Non-finite source-ablation gradient')
                if p.grad is None:
                    p.grad = value.detach().clone()
                else:
                    p.grad.add_(value.detach())
        for label in (0, 1):
            ix = labels == label
            self.pairs[label] += int(ix.sum())
            self.counts[label] += int(counts[ix].sum())
            self.losses[label] += float(sums[ix].detach().double().sum())

    def record(self):
        if sum(self.counts) != self.denominator:
            raise RuntimeError('Ablation batch denominator changed')
        return {'denominator': self.denominator, 'real_pairs': self.pairs[0], 'synthetic_pairs': self.pairs[1],
                'real_weighted_loss': self.losses[0] / self.denominator,
                'synthetic_weighted_loss': self.losses[1] / self.denominator}


class EpochObserver:
    def __init__(self, root, c, probe, binding, row_mapping):
        self.root, self.c, self.probe, self.binding = Path(root), c, probe, binding
        self.identity = digest({'probe': probe['identity'], 'binding': binding})
        self.row_mapping = np.asarray(row_mapping)
        self.rows, self.scores = [], []

    def _score(self, model):
        from .engine import evaluate
        modes = [(m, m.training) for m in model.modules()]
        try:
            with preserve_rng():
                result = evaluate(model, self.probe['tokens'], self.c,
                                  masks=self.probe['attention'], loss_masks=self.probe['scoring'])
            return result['loss'], result['perplexity']
        finally:
            for module, mode in modes:
                module.training = mode

    def start(self, model, global_step):
        self.start_step = global_step
        uncommitted=self.root/'epoch_diagnostics_manifest.json'
        if uncommitted.exists():
            previous=read_json(uncommitted)
            if previous['identity']!=self.identity or previous['binding']!=self.binding:
                raise RuntimeError('Interrupted branch settings changed')
            verify_files(self.root,previous['files'])
            archive=(self.root/previous['records_path']).parent/'uncommitted_seal.json'
            if archive.exists():
                raise RuntimeError('Uncommitted branch seal already has an archive')
            uncommitted.replace(archive)
        self.attempt = self.root / 'epoch_diagnostics' / uuid.uuid4().hex
        self.attempt.mkdir(parents=True, exist_ok=False)
        atomic_json(self.attempt / 'attempt.json', {'identity': self.identity, 'binding': self.binding,
                                                  'start_step': global_step, 'complete': False})
        self.before, self.before_ppl = self._score(model)
        self.initial = {'nll': self.before, 'perplexity': self.before_ppl}

    def after_step(self, model, *, step, batch_indices, **metrics):
        started=time.perf_counter()
        nll, ppl = self._score(model)
        row = {'step': step, 'local_step': step-self.start_step, 'probe_nll_before': self.before,
               'probe_nll': nll, 'probe_perplexity': ppl, 'source': self.binding['source'],
               'original_rows': self.row_mapping[batch_indices].tolist(),
               'probe_seconds':time.perf_counter()-started, **metrics}
        if self.binding['source'] != 'both' and not metrics[self.binding['source'] + '_pairs']:
            if metrics['gradient_norm'] != 0 or nll != self.before:
                raise RuntimeError('Empty-source update moved weights or predictions')
        self.rows.append(row)
        with (self.attempt / 'updates.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        self.before, self.before_ppl = nll, ppl

    def seal(self, terminal):
        if not self.rows or [x['local_step'] for x in self.rows] != list(range(1, len(self.rows)+1)):
            raise RuntimeError('Incomplete branch journal')
        result = {'identity': self.identity, 'binding': self.binding, 'terminal_weights_sha256': terminal,
                  'updates': len(self.rows), 'initial_probe': self.initial,
                  'records_path': (self.attempt/'updates.jsonl').relative_to(self.root).as_posix(),
                  'files': {p.relative_to(self.root).as_posix(): sha256(p) for p in self.attempt.iterdir()}}
        atomic_json(self.root / 'epoch_diagnostics_manifest.json', result)
        return result

    def verify(self, terminal):
        result = read_json(self.root / 'epoch_diagnostics_manifest.json')
        if not result or result['identity'] != self.identity or result['terminal_weights_sha256'] != terminal:
            raise RuntimeError('Completed branch diagnostics differ')
        verify_files(self.root, result['files'])
        rows=[json.loads(line) for line in (self.root/result['records_path']).read_text().splitlines()]
        expected=math.ceil(len(self.row_mapping)/self.c['effective_batch'])
        if (len(rows)!=expected or result['updates']!=expected
                or [x['local_step'] for x in rows]!=list(range(1,expected+1))
                or any(x['source']!=self.binding['source'] for x in rows)):
            raise RuntimeError('Completed branch update timeline differs')
        return result
