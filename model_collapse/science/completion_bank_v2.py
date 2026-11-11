"""Versioned native-tokenizer bank, preserving historical document assignments."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from filelock import FileLock

from ..common import PROJECT, STATE, atomic_json, digest, read_json, sha256, verify_files
from ..completion_data import _pairs, _write_collection, records, load_pairs
from ..fresh_data import document_keys, passage_hashes, encode_sequences
from ..data import save_array


def bank_contract(c):
    keys = ("model", "model_revision", "dataset", "dataset_revision", "dataset_config", "dataset_files",
            "seed", "shuffle_buffer", "full_cycles", "full_train_pairs", "corpus_bank", "source_opt_bank_manifest_sha256")
    return digest({"version": 2, "settings": {k:c[k] for k in keys}, "prompt":64, "target":64,
                   "implementation": sha256(__file__), "original_helper":sha256(PROJECT/'model_collapse/completion_data.py')})


def stream(c, coverage):
    from datasets import load_dataset
    files = list(c["dataset_files"]);random.Random(c["seed"]).shuffle(files)
    coverage.update(shard_order=files, read_rows_by_file={})
    def source():
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip('/')
        for filename in files:
            url = f"{endpoint}/datasets/{c['dataset']}/resolve/{c['dataset_revision']}/{filename}"
            for index,row in enumerate(load_dataset('parquet', data_files=url, split='train', streaming=True)):
                coverage['read_rows_by_file'][filename]=index+1
                yield {**row,'source_file':filename,'source_row':index}
    rng=random.Random(c['seed']+1);buffer=[]
    for row in source():
        if len(buffer)<c['shuffle_buffer']:buffer.append(row)
        else:
            index=rng.randrange(len(buffer));yield buffer[index];buffer[index]=row
    rng.shuffle(buffer);yield from buffer


def prepare_bank(root,c,tokenizer,*,source=None,original=None,iterator=None,source_tokenizer=None):
    from ..completion_data import source_bank
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    original=Path(original) if original else PROJECT/c['source_opt_bank']
    if not (original/'manifest.json').exists():
        from ..completion_data import restore_bank
        restore_bank(c['source_opt_bank_ref'],original)
    if sha256(original/'manifest.json')!=c['source_opt_bank_manifest_sha256']:
        raise RuntimeError('Original document allocation bank differs')
    original_manifest=read_json(original/'manifest.json')
    verify_files(original,{name:value for name,value in original_manifest['files'].items()
                          if name.endswith('/documents.jsonl') and name.startswith('cycle_')})
    source=Path(source) if source else source_bank(c)[0]
    contract=bank_contract(c)
    with FileLock(str(root/'prepare.lock'),timeout=0):
        if (old:=read_json(root/'manifest.json')) is not None:
            if old['contract']!=contract:raise RuntimeError('Native bank contract changed; use a new directory')
            verify_files(root,old['files']);return old
        prior=read_json(root/'preparation.json')
        if prior and prior['contract']!=contract:raise RuntimeError('Native preparation contract changed')
        atomic_json(root/'preparation.json',{'contract':contract})
        seen=set();spans=set();collections={};coverage={};retained={};rejected={'identity':0,'passage':0,'short':0}
        # All held-out identities/passages, including secondary corpora, are reserved first.
        for path in sorted(source.glob('*/*/documents.jsonl')):
            if path.parent.name not in ('validation','test'):continue
            for row in records(path):
                seen.update(document_keys(row));spans.update(passage_hashes(tokenizer.encode(row['text'],add_special_tokens=False)))
        for split in ('validation','test'):
            rows=list(records(source/'fineweb_edu'/split/'documents.jsonl'))
            values=((row,tokenizer.encode(row['text'],add_special_tokens=False)) for row in rows)
            docs,windows,prompts,targets=_pairs(values)
            collections[split]=_write_collection(root/split,docs,windows,prompts,targets,np.arange(len(prompts)))
        originals={f'cycle_{i:02d}':list(records(original/f'cycle_{i:02d}/documents.jsonl')) for i in range(1,c['full_cycles']+1)}
        reserved_original=set().union(*(document_keys(row) for rows in originals.values() for row in rows))
        iterator=iter(iterator) if iterator is not None else iter(stream(c,coverage))
        def accepted(row,*,extension=False):
            keys=document_keys(row)
            if not keys or seen.intersection(keys) or (extension and reserved_original.intersection(keys)):
                rejected['identity']+=1;return None
            seen.update(keys)
            ids=tokenizer.encode(row['text'],add_special_tokens=False);fingerprints=passage_hashes(ids)
            if spans.intersection(fingerprints):rejected['passage']+=1;return None
            if len(ids)<128:rejected['short']+=1;return None
            spans.update(fingerprints)
            return row,ids
        try:
            for cycle in range(1,c['full_cycles']+1):
                name=f'cycle_{cycle:02d}';destination=root/name
                # Consume accepted original documents in their original order, then deterministic extensions.
                added=[]
                def eligible():
                    for row in originals[name]:
                        value=accepted(row)
                        if value is not None:yield value
                    for row in iterator:
                        value=accepted(row,extension=True)
                        if value is not None:
                            added.append({'id':row.get('id'),'source_file':row.get('source_file'),'source_row':row.get('source_row')})
                            yield value
                docs,windows,prompts,targets=_pairs(eligible(),c['full_train_pairs'])
                if len(prompts)!=c['full_train_pairs']:raise RuntimeError('Native source exhausted before filling chunks')
                order=np.random.default_rng(c['seed']+cycle).permutation(len(prompts))
                collections[name]=_write_collection(destination,docs,windows,prompts,targets,order)
                retained[name]={'original_documents_available':len(originals[name]),'selected_documents':len(docs),'added_documents':added}
                print(f'Prepared {name}: {len(prompts)} pairs, {len(docs)} documents, {len(added)} extensions',flush=True)
        finally:
            close=getattr(iterator,'close',None)
            if close:close()
        # Keep the exact historical selected text spans for secondary evaluation,
        # translating their OPT offsets before native tokenization.
        from transformers import AutoTokenizer
        opt=source_tokenizer
        if opt is None:
            try:
                opt=AutoTokenizer.from_pretrained('facebook/opt-125m',revision='27dcfa74d334bc871f3234de431e71c6eeba5dd6',local_files_only=True)
            except OSError:
                opt=AutoTokenizer.from_pretrained('facebook/opt-125m',revision='27dcfa74d334bc871f3234de431e71c6eeba5dd6')
        secondary={}
        for corpus in c['secondary_corpora']:
            sequences=[];selected=[]
            for row in records(source/corpus/'test/documents.jsonl'):
                old_ids=opt.encode(row['text'],add_special_tokens=False)
                start=row.get('window_start',0);end=start+row['used_source_tokens']
                text=opt.decode(old_ids[start:end],clean_up_tokenization_spaces=False)
                ids=tokenizer.encode(text,add_special_tokens=False)
                if not ids:continue
                sequences.extend(encode_sequences(ids,tokenizer,512,genuine_end=row.get('genuine_document_end',True)))
                selected.append({'id':row.get('id'),'original_selected_tokens_sha256':row.get('selected_tokens_sha256'),
                                 'native_tokens':len(ids),'text_sha256':__import__('hashlib').sha256(text.encode()).hexdigest()})
            maximum=max(map(len,sequences));tokens=np.full((len(sequences),maximum),tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,dtype=np.int32)
            masks=np.zeros_like(tokens)
            for i,seq in enumerate(sequences):tokens[i,:len(seq)]=seq;masks[i,:len(seq)]=1
            destination=root/'secondary'/corpus
            save_array(destination/'tokens.npy',tokens);save_array(destination/'masks.npy',masks)
            atomic_json(destination/'sources.json',selected)
            secondary[corpus]={'sequences':len(tokens),'scored_targets':int(masks[:,1:].sum())}
        windows=np.load(root/'validation/windows.npy');by_doc={}
        for i,(doc,_) in enumerate(windows):by_doc.setdefault(int(doc),[]).append(i)
        rng=random.Random(c['seed']);chosen=rng.sample(sorted(by_doc),256)
        save_array(root/'probe_indices.npy',np.array([rng.choice(by_doc[d]) for d in chosen],dtype=np.int32))
        newline=[i for i in range(len(tokenizer)) if '\n' in tokenizer.decode([i],clean_up_tokenization_spaces=False)]
        examples=[]
        for name in ('cycle_01','validation','test'):
            p,t,_=load_pairs(root,name,3)
            examples.extend({'collection':name,'index':i,'prompt':tokenizer.decode(p[i]),'real_continuation':tokenizer.decode(t[i])} for i in range(3))
        atomic_json(root/'examples.json',examples);atomic_json(root/'allocation.json',retained)
        files={p.relative_to(root).as_posix():sha256(p) for p in root.rglob('*') if p.is_file() and p.name not in ('manifest.json','prepare.lock') and not p.name.endswith('.partial')}
        files.update({f'{name}/manifest.json':sha256(root/name/'manifest.json') for name in collections})
        result={'version':2,'contract':contract,'collections':collections,'secondary':secondary,'files':files,
            'bos_token_id':tokenizer.bos_token_id,'eos_token_id':tokenizer.eos_token_id,'newline_token_ids':newline,
            'coverage':coverage,'rejected':rejected,'source_opt_bank_manifest_sha256':c['source_opt_bank_manifest_sha256'],
            'verification_scope':'Native-token document identity and exact sliding64 passage exclusion; no claim of pretraining novelty.'}
        atomic_json(root/'manifest.json',result);return result


def publish_bank(root,c,repos):
    from huggingface_hub import HfApi,get_token
    from ..hub import resolve_snapshot,retry_transfer
    root=Path(root);manifest=read_json(root/'manifest.json');verify_files(root,manifest['files'])
    name='smollm2-completion64-bank-'+manifest['contract'][:16]
    files={p:{'sha256':h,'bytes':(root/p).stat().st_size,'repo_type':'dataset'} for p,h in {**manifest['files'],'manifest.json':sha256(root/'manifest.json')}.items()}
    atomic_json(root/'artifact_index.json',{'run_id':name,'repos':repos,'files':files})
    api=HfApi(token=os.environ.get('HF_TOKEN') or get_token())
    if not api.repo_info(repos['dataset'],repo_type='dataset').private:raise RuntimeError('Native bank backup must be private')
    retry_transfer(api.upload_folder,repo_id=repos['dataset'],repo_type='dataset',folder_path=root,path_in_repo=name,allow_patterns=[*files,'artifact_index.json'],commit_message='Freeze SmolLM2 native completion bank')
    ref={'run_id':name,'repos':repos,'revisions':resolve_snapshot(repos),'artifact_index_sha256':sha256(root/'artifact_index.json'),
         'manifest_sha256':sha256(root/'manifest.json'),'contract':bank_contract(c)}
    atomic_json(STATE/'smollm2_completion64_bank.json',ref);return ref
