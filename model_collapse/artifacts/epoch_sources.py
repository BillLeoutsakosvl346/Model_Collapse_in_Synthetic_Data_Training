"""Pinned selective source recovery for full-epoch SGD source ablations."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..common import PROJECT, atomic_json, read_json, sha256, verify_files
from ..completion_data import assemble
from ..completion_mixing import select_rows
from .gradient_sources import Snapshot, check_input


class EpochSnapshot(Snapshot):
    """Read retained working bytes only when they match the pinned backup index."""
    def __init__(self,ref,root,store=None):
        self.store=store
        self.connection=None
        super().__init__(ref,root)

    def get(self,relative):
        meta=self.index.get(relative)
        path=self.root/relative
        if (self.store and meta and 'storage' not in meta
                and (not path.is_file() or sha256(path)!=meta['sha256'])):
            from ..cloud.s3_artifacts import client
            import hashlib
            from botocore.exceptions import ClientError
            try:
                self.connection=self.connection or client(self.store)
                raw=self.connection.get_object(Bucket=self.store['bucket'],
                    Key=self.store['prefix']+'/'+self.ref['run_id']+'/'+relative)['Body'].read()
            except ClientError as exc:
                if str(exc.response['Error']['Code']) not in ('404','NoSuchKey','NotFound'):
                    raise
            else:
                if len(raw)!=meta['bytes'] or hashlib.sha256(raw).hexdigest()!=meta['sha256']:
                    raise RuntimeError('Retained source differs from pinned backup: '+relative)
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_name(path.name+'.partial');temporary.write_bytes(raw);temporary.replace(path)
        return super().get(relative)


def seed_cache(root, c):
    """Copy only verified local evidence into a new non-authoritative cache."""
    root = Path(root)
    local = PROJECT / '.state/sgd-update-analysis'
    refs = [('parent', c['source_parent'])]
    old = local / c['source_parent']['run_id']
    if not (old/'child_snapshots.json').exists():
        return
    children = read_json(old/'child_snapshots.json')
    scientific = read_json(old/'config.json')
    refs += [('mixed99', children['mixed99']), ('bank', scientific['prepared_bank'])]
    for label, ref in refs:
        origin = PROJECT/'data/smollm2_completion64_v2' if label == 'bank' else local/ref['run_id']
        dest = root/'.sources'/label
        dest.mkdir(parents=True, exist_ok=True)
        ix = origin/'artifact_index.json'
        if not ix.exists() and label == 'bank':
            ix = PROJECT/'.state/sgd-bank-source'/'artifact_index.json'
        if ix.exists() and sha256(ix) == ref['artifact_index_sha256']:
            shutil.copyfile(ix, dest/'artifact_index.json')
        if not (dest/'artifact_index.json').exists():
            continue
        for rel, meta in read_json(dest/'artifact_index.json')['files'].items():
            src = origin/rel
            if src.is_file() and src.suffix not in ('.pt', '.safetensors') and sha256(src) == meta['sha256']:
                (dest/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest/rel)


def restore(root, c, *, weights=True, cycles=None, replay_endpoint=True):
    root = Path(root)
    parent = EpochSnapshot(c['source_parent'], root/'.sources/parent',c.get('source_store'))
    scientific = parent.json('config.json')
    if sha256(parent.root/'config.json') != c['source_config_sha256']:
        raise RuntimeError('Historical source configuration changed')
    child = EpochSnapshot(parent.json('child_snapshots.json')[c['source_arm']], root/'.sources/mixed99',c.get('source_store'))
    native = child.json('config.json')
    if (native['optimizer'] != 'sgd' or native['learning_rate'] != .01 or native['precision'] != 'fp32'
            or native['synthetic_fraction'] != .99 or native['generation_decoder'] != 'greedy'):
        raise RuntimeError('Expected the completed 99% SGD/greedy condition')
    reference = root/'reference'
    reference.mkdir(exist_ok=True)
    for i in range(11):
        original=child.get(f'seed_42/gen_{i:02d}/metrics.json')
        shutil.copyfile(original,reference/f'metrics_{i:02d}.json')
    bank = Snapshot(scientific['prepared_bank'], root/'.sources/bank')
    bm = bank.json('manifest.json')
    if sha256(bank.root/'manifest.json') != scientific['prepared_bank']['manifest_sha256']:
        raise RuntimeError('FineWeb bank changed')
    native['bos_token_id'] = bm['bos_token_id']
    bank.get('probe_indices.npy')
    evaluations = {}
    for split in ('validation', 'test'):
        prompts, targets = (bank.array(split+'/'+name+'.npy') for name in ('prompts', 'targets'))
        bank.get(split+'/windows.npy'); bank.get(split+'/documents.jsonl'); bank.get(split+'/manifest.json')
        evaluations[split] = assemble(prompts, targets, bm['bos_token_id'])
    from ..workflow.completion import sgd_update_probe
    probe = sgd_update_probe(bank.root, native)
    cases = []
    for cycle in cycles or c['cycles']:
        prefix, prior = f'seed_42/gen_{cycle:02d}', f'seed_42/gen_{cycle-1:02d}'
        old_training = child.json(prefix+'/training.json')
        initial = child.json(prior+'/training.json')
        before_metrics, after_metrics = child.json(prior+'/metrics.json'), child.json(prefix+'/metrics.json')
        prior_seal = child.json(prior+'/stage_completion.json')
        terminal_seal = child.json(prefix+'/stage_completion.json')
        checkpoint = child.checkpoint(prior, restore_weights=weights)
        expected_checkpoint = child.checkpoint(prefix, restore_weights=weights and replay_endpoint)
        state_name = prior+'/optimizer_state/cycle.pt'
        state = child.get(state_name) if weights else child.root/state_name
        for name, record in ((prior, initial), (prefix, old_training)):
            for rel, value in {**{'checkpoint/'+k:v for k,v in record['checkpoint_files'].items()}, **record['state_files']}.items():
                checksum = value['sha256'] if isinstance(value, dict) else value
                if child.index[name+'/'+rel]['sha256'] != checksum:
                    raise RuntimeError('Historical checkpoint/state seal differs')
            if weights and (name == prior or replay_endpoint):
                verify_files(child.root/name, {'checkpoint/'+k:v for k,v in record['checkpoint_files'].items()})
        for rel in ('checkpoint/model.safetensors', 'optimizer_state/cycle.pt'):
            if child.index[prior+'/'+rel]['sha256'] != prior_seal['files'][rel]:
                raise RuntimeError('Initializer completion differs')
        if child.index[prefix+'/checkpoint/model.safetensors']['sha256'] != terminal_seal['files']['checkpoint/model.safetensors']:
            raise RuntimeError('Expected replay endpoint differs')
        initializer = initial['selected_weights_sha256']
        if (old_training['initial_weights_sha256'] != initializer
                or old_training['optimizer_parent_sha256'] != child.index[state_name]['sha256']):
            raise RuntimeError('Epoch initializer/optimization parent changed')
        inp = child.json(prefix+'/training_input/manifest.json')
        tokens, attention, scoring = (child.array(prefix+'/training_input/'+name+'.npy') for name in ('tokens','attention','loss_mask'))
        verify_files(child.root/prefix/'training_input', inp['files'])
        prompts, real = (bank.array(f'cycle_{cycle:02d}/'+n+'.npy') for n in ('prompts','targets'))
        bank.get(f'cycle_{cycle:02d}/manifest.json')
        selection, real_count = select_rows(10000, .99, 42, cycle)
        mixture = child.json(prefix+'/mixture/manifest.json')
        for name in ('permutation','synthetic_indices','source_labels','row_indices','targets'):
            actual = child.array(prefix+'/mixture/'+name+'.npy')
            if name in selection and not np.array_equal(actual, selection[name]):
                raise RuntimeError('Historical source selection changed')
        child.get(prefix+'/mixture/selection.json')
        verify_files(child.root/prefix/'mixture', mixture['files'])
        generation = child.json(f'synthetic_for_gen_{cycle:02d}/manifest.json')
        generated = child.array(f'synthetic_for_gen_{cycle:02d}/tokens.npy')
        if (real_count != 100 or mixture['generator_weights_sha256'] != initializer
                or inp['lineage']['generation_manifest_sha256'] != sha256(child.root/f'synthetic_for_gen_{cycle:02d}/manifest.json')
                or inp['lineage']['mixture_manifest_sha256'] != sha256(child.root/prefix/'mixture/manifest.json')
                or generation['provenance']['parent_weights_sha256'] != initializer):
            raise RuntimeError('Saved target generator/mixture provenance differs')
        check_input(tokens, attention, scoring, prompts, real, generated, selection['synthetic_indices'], inp,
                    initializer, sha256(bank.root/'manifest.json'), sha256(bank.root/f'cycle_{cycle:02d}/manifest.json'))
        diag = child.json(prefix+'/diagnostics_manifest.json')
        for rel in diag['files']:
            child.get(prefix+'/'+rel)
        verify_files(child.root/prefix, diag['files'])
        if diag['probe_identity'] != probe['identity']:
            raise RuntimeError('Historical real-validation probe changed')
        records = [__import__('json').loads(line) for line in (child.root/prefix/diag['records_path']).read_text().splitlines()]
        shutil.copyfile(child.root/prefix/diag['records_path'],reference/f'updates_{cycle:02d}.jsonl')
        cases.append({'cycle': cycle, 'tokens': tokens, 'attention': attention, 'scoring': scoring,
                      'labels': selection['source_labels'], 'checkpoint': checkpoint, 'expected_checkpoint': expected_checkpoint,
                      'state': state, 'initial_hash': initializer, 'training': old_training, 'historical_records': records,
                      'historical_lineage': inp['lineage'], 'before_metrics': before_metrics, 'after_metrics': after_metrics,
                      'terminal_hash': old_training['selected_weights_sha256'], 'native': native})
    atomic_json(root/'source_recovery.json', {'parent':parent.receipt(), 'child':child.receipt(), 'bank':bank.receipt(),
                                            'bulk_restored':weights, 'cycles':cycles or c['cycles']})
    atomic_json(reference/'manifest.json',{'files':{p.name:sha256(p) for p in reference.iterdir() if p.is_file() and p.name!='manifest.json'},
        'source_files':{**{f'metrics_{i:02d}.json':f'seed_42/gen_{i:02d}/metrics.json' for i in range(11)},
                        **{f'updates_{case["cycle"]:02d}.jsonl':f'seed_42/gen_{case["cycle"]:02d}/'+child.json(f'seed_42/gen_{case["cycle"]:02d}/diagnostics_manifest.json')['records_path'] for case in cases}},
        'source_snapshot':child.ref})
    return cases, evaluations, probe, native
