"""Publish a native bank through existing private S3 objects and pinned Hub indexes."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..common import STATE, atomic_json, read_json, sha256, verify_files


def publish_bank(root,c,repos):
    from huggingface_hub import HfApi,get_token
    from .hub import resolve_snapshot,retry_transfer
    from ..cloud.s3_artifacts import BULK_SUFFIXES,client,upload,check_capacity
    from ..science.completion_bank_v2 import bank_contract
    root=Path(root);manifest=read_json(root/'manifest.json')
    if manifest['contract'] != bank_contract(c):
        raise RuntimeError('Native builder changed; retain this bank and prepare a new directory')
    verify_files(root,manifest['files'])
    store=read_json(STATE/'smollm2-bank-storage.json')
    if not store:raise RuntimeError('Configure private native-bank backup storage before publishing')
    api=HfApi(token=os.environ.get('HF_TOKEN') or get_token())
    if not api.repo_info(repos['dataset'],repo_type='dataset').private:
        raise RuntimeError('Native bank metadata must be private')
    connection=client(store,connect_timeout=5)
    from ..runpod_api import Runpod
    volume=Runpod().request('GET','/network-volumes/'+store['bucket'])
    if volume['dataCenter']!=store['region']:raise RuntimeError('Native bank storage region differs')
    check_capacity(store,volume['size'],sum((root/p).stat().st_size for p in manifest['files']),api=connection)
    name='smollm2-completion64-bank-'+manifest['contract'][:16]
    files={p:{'sha256':h,'bytes':(root/p).stat().st_size,'repo_type':'dataset'}
           for p,h in {**manifest['files'],'manifest.json':sha256(root/'manifest.json')}.items()}
    def transfer(item):
        relative,meta=item
        meta['storage']=upload(root/relative,meta,store,connection=connection)
        print('Backed up native bank:',relative,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(transfer,[(p,m) for p,m in files.items() if p.endswith(BULK_SUFFIXES)]))
    atomic_json(root/'artifact_index.json',{'run_id':name,'repos':repos,'files':files})
    paths=[p for p,meta in files.items() if 'storage' not in meta]+['artifact_index.json']
    retry_transfer(api.upload_folder,repo_id=repos['dataset'],repo_type='dataset',folder_path=root,
                   path_in_repo=name,allow_patterns=paths,commit_message='Freeze native SmolLM2 bank with private bulk recovery')
    ref={'run_id':name,'repos':repos,'revisions':resolve_snapshot(repos),
         'artifact_index_sha256':sha256(root/'artifact_index.json'),'manifest_sha256':sha256(root/'manifest.json'),
         'contract':manifest['contract'],'bulk_store':store}
    atomic_json(STATE/'smollm2_completion64_bank.json',ref)
    return ref
