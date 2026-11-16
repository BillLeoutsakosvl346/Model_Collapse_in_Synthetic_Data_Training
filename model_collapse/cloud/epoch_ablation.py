"""Guarded deployment and operations for the bounded epoch-ablation matrix."""
from __future__ import annotations

import json
import os
import shlex
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from ..common import PROJECT, STATE, atomic_json, config, digest, load_secrets, now, read_json, sha256
from ..runpod_api import Runpod
from ..workflow.epoch_ablation import ALLOCATION
from . import execution as cloud


def validation_deadline(created, checked):
    if checked['previously_accepted']:
        return created+checked['remaining_seconds']
    return created+min(900,checked['remaining_validation_usd']/checked['guard_hourly_usd']*3600)-60


def session_name(run_id, launch_no):
    return 'iib-'+run_id+f'-{launch_no:02d}'


def input_archive(source_cache):
    """Reuse only a content-verified bundle, avoiding recompression during stock races."""
    source_cache=Path(source_cache)
    files=sorted(p for p in source_cache.rglob('*') if p.is_file())
    inventory=[{'path':p.relative_to(source_cache).as_posix(),'sha256':sha256(p),
                'bytes':p.stat().st_size} for p in files]
    identity=digest(inventory)
    archive=STATE/('epoch-ablation-inputs-'+identity+'.tar.gz')
    receipt=archive.with_suffix('.json')
    previous=read_json(receipt) or {}
    if (archive.is_file() and previous.get('inventory_sha256')==identity
            and previous.get('bytes')==archive.stat().st_size
            and previous.get('sha256')==sha256(archive)):
        return archive
    temporary=archive.with_suffix('.tmp')
    with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
        for p in files:
            tar.add(p,arcname=p.relative_to(source_cache).as_posix())
    # A concurrent source edit must never be certified as the earlier inventory.
    if any(p.stat().st_size!=item['bytes'] or sha256(p)!=item['sha256']
           for p,item in zip(files,inventory)):
        raise RuntimeError('Input cache changed while packaging; retry from verified sources')
    os.replace(temporary,archive)
    atomic_json(receipt,{'inventory_sha256':identity,'sha256':sha256(archive),
                         'bytes':archive.stat().st_size})
    return archive


def archive_controls(sftp, remote, launch_id):
    """Preserve previous-launch controls before a fresh independent watchdog starts."""
    parent=remote+'/control_history'
    try:sftp.mkdir(parent)
    except OSError:
        sftp.stat(parent)
    dest=parent+'/'+launch_id
    sftp.mkdir(dest)
    for name in ('stop_request.json','supervisor.json','watchdog.json','watchdog_exit.json',
                 'status.json','resources.json','validation_guard_deadline.json','pipeline.log','supervisor.log'):
        source=remote+'/'+name
        try:sftp.stat(source)
        except FileNotFoundError:continue
        sftp.rename(source,dest+'/'+name)


def record_volume(volume, run_id, created, size):
    from filelock import FileLock
    with FileLock(str(STATE/'ledger.lock'),timeout=30):
        d=cloud.ledger();previous=d.setdefault('volumes',{}).get(volume['id'])
        if previous and previous['run_id']!=run_id:
            raise RuntimeError('Volume ledger ownership differs')
        d['volumes'].setdefault(volume['id'],{'run_id':run_id,'allocation':ALLOCATION,
            'volume_id':volume['id'],'charge_started_at':created,'hourly_usd':size*.07/730,'kind':'diagnostic'})
        atomic_json(STATE/'ledger.json',d)


def create_volume(api, local, run_id, name, size, dc, role):
    request={'name':name,'size':size,'dataCenter':dc,'type':'STANDARD'};started=now()
    if any(v.get('name')==name for v in api.request('GET','/network-volumes')['networkVolumes']):
        raise RuntimeError('A volume with this exact new-launch name already exists')
    atomic_json(local/('pending_'+role+'_volume.json'),{'request':request,'created_at':started})
    try:
        volume=api.request('POST','/network-volumes',request)
    except Exception:
        matches=[v for v in api.request('GET','/network-volumes')['networkVolumes'] if v.get('name')==name]
        if len(matches)!=1:
            raise
        volume=matches[0]
    atomic_json(local/('created_'+role+'_volume.json'),volume)
    record_volume(volume,run_id,started,size)
    return volume


def remote_state(resources):
    """Read task-owned working storage without starting compute or trusting local caches."""
    from .s3_artifacts import client
    from botocore.exceptions import ClientError
    store={'backend':'runpod_s3','bucket':resources['volume_id'],'region':resources['data_center'],'prefix':'runs'}
    api=client(store);out={}
    for name in ('config.json','completion.json','backup_snapshot.json','artifact_index.json','acceptance.json'):
        try:
            raw=api.get_object(Bucket=store['bucket'],Key='runs/'+resources['run_id']+'/'+name)['Body'].read()
        except ClientError as exc:
            if str(exc.response['Error']['Code']) not in ('404','NoSuchKey','NotFound'):
                raise
            continue
        out[name]=json.loads(raw)
    if out.get('config.json') and digest(out['config.json'])!=resources['configuration_sha256']:
        raise RuntimeError('Remote diagnostic configuration differs from authoritative ownership')
    accepted=out.get('acceptance.json')
    if accepted:
        if accepted['runtime_sha256']!=resources['code_sha256'] or accepted['configuration_sha256']!=resources['configuration_sha256']:
            raise RuntimeError('Recovered GPU acceptance has a different identity')
        atomic_json(PROJECT/'runs'/resources['run_id']/'acceptance.json',accepted)
    return out


def _used(state):
    launches=[x for x in state['launches'] if x.get('allocation')==ALLOCATION]
    ids={x['volume_id'] for x in launches}
    selected={**state,'launches':launches,'volumes':{k:v for k,v in state.get('volumes',{}).items() if k in ids or v.get('allocation')==ALLOCATION}}
    seconds=sum(max(0,x.get('compute_stopped_at',x.get('stopped_at',now()))-x['created_at']) for x in launches)
    validation=sum(x.get('validation_estimated_usd',0) for x in launches)
    return launches,cloud.accrued(selected),seconds,validation


def preflight(c, resume_id=None):
    load_secrets();api=Runpod();account=api.account();state=cloud.ledger()
    if account['isAutoPayEnabled'] or account['enableBackupPayments']:
        raise RuntimeError('Automatic and backup payments must remain disabled')
    allocation=state.get('allocations',{}).get(ALLOCATION,{})
    if (allocation.get('budget_usd')!=4 or allocation.get('reserve_usd')!=2
            or cloud.task_budget(state)<130 or cloud.setup_budget(state)<6.10):
        raise RuntimeError('Approved epoch-ablation limits are not recorded')
    launches,used,seconds,validation=_used(state)
    if launches and not resume_id:
        raise RuntimeError('This diagnostic already owns a launch; inspect/resume its ID')
    if resume_id and not any(x['run_id']==resume_id for x in launches):
        raise RuntimeError('Resume does not own this task allocation')
    cloud.check_active_pods(c,api.request('GET','/pods')['pods'],state)
    gpu=api.request('GET','/catalog/gpus/'+quote(c['required_gpu'],safe='')+'?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=13.2')
    choices=cloud.configured_placements(c,[{'gpu':c['required_gpu'],'data_center':d['id']} for d in gpu.get('dataCenters',[]) if d.get('availability') not in ('NONE','UNAVAILABLE')])
    if not choices:
        raise RuntimeError('MATCHING_GPU_UNAVAILABLE: no compatible RTX PRO 4000 capacity; nothing provisioned')
    hourly=gpu['price']['secure'];guard=(hourly+20*.10/730+40*.07/730)*1.15
    remaining_seconds=5400-seconds
    retained=40*.07/730*24*7
    forecast=guard*remaining_seconds/3600+retained+cloud.historical_running_storage(state,remaining_seconds/3600)
    available=min(4-used-2,130-cloud.accrued(state)-2,account['clientBalance']-2)
    if available<forecast or remaining_seconds<300:
        raise RuntimeError(f'Diagnostic allowance cannot cover deadline/storage: available {available:.3f}, forecast {forecast:.3f}')
    accepted=bool(resume_id and read_json(PROJECT/'runs'/resume_id/'acceptance.json'))
    allowance=min(.15-validation,cloud.setup_budget(state)-cloud.accrued(state,'smoke'))
    if not accepted and allowance<.03:
        raise RuntimeError('Insufficient remaining acceptance/setup allowance')
    # Validate pinned small inputs on CPU; bulk weights stay in cloud storage.
    from ..artifacts.epoch_sources import seed_cache,restore
    owner=cloud.local_resources(c['source_parent']['run_id'])
    if owner['volume_id']!=c['source_store']['bucket']:
        raise RuntimeError('Historical retained source does not match its authoritative registry')
    cache=STATE/'epoch-ablation-inputs';seed_cache(cache,c)
    cases,_,_,_=restore(cache,c,weights=False)
    from .s3_artifacts import client,validate_reference
    from ..artifacts.gradient_sources import Snapshot
    parent=Snapshot(c['source_parent'],cache/'.sources/parent')
    child=Snapshot(parent.json('child_snapshots.json')['mixed99'],cache/'.sources/mixed99')
    for cycle in (2,3,5,6,9,10):
        rel=f'seed_42/gen_{cycle:02d}/checkpoint/model.safetensors';meta=child.index[rel]
        store=validate_reference(meta['storage'],meta['sha256'])
        if client(store).head_object(Bucket=store['bucket'],Key=meta['storage']['key'])['ContentLength']!=meta['bytes']:
            raise RuntimeError('Historical checkpoint is absent or has changed size')
    return {'account_balance_usd':account['clientBalance'],'hourly_usd':hourly,'guard_hourly_usd':guard,
            'data_center':choices[0]['data_center'],'remaining_seconds':remaining_seconds,
            'remaining_validation_usd':allowance,'remaining_task_usd':available,
            'forecast_usd':forecast,'retained_seven_days_usd':retained,'input_cache':str(cache),
            'repos':c['source_parent']['repos'],'previously_accepted':accepted}


def _observe(resources):
    api=Runpod();pod=api.pod(resources['pod_id'])
    if pod['status']!='RUNNING':
        return {'pod_state':pod['status']}
    client=cloud.ssh_connect(pod,STATE/'runpod_ed25519')
    try:
        script="import json,pathlib; p=pathlib.Path("+repr(resources['remote_root'])+"); print(json.dumps({k:json.loads((p/(k+'.json')).read_text()) if (p/(k+'.json')).exists() else None for k in ['status','supervisor','watchdog','acceptance','completion','coordinator']}))"
        return json.loads(cloud.command(client,'python3 -c '+shlex.quote(script),timeout=20))
    finally:
        client.close()


def verify_detached(run_id, timeout=1800):
    r=cloud.local_resources(run_id);limit=min(now()+timeout,r['absolute_deadline']-15)
    while now()<limit:
        first=_observe(r);s=first.get('status') or {}
        if s.get('state')=='failed':
            raise RuntimeError('Epoch diagnostic failed: '+s.get('error','unknown failure'))
        accepted=first.get('acceptance')
        if accepted and s.get('phase')=='full' and s.get('local_step',0)>=2:
            if accepted['runtime_sha256']!=r['code_sha256']:
                raise RuntimeError('Acceptance runtime differs from deployed source')
            atomic_json(PROJECT/'runs'/run_id/'acceptance.json',accepted)
            time.sleep(5)  # Fresh SSH session; training must survive disconnection.
            second=_observe(r);s2=second.get('status') or {}
            healthy=all(second.get(k) and now()-second[k]['time']<20
                        and second[k]['time']>first[k]['time'] for k in ('supervisor','watchdog'))
            progressed=(s2.get('cycle',0),s2.get('local_step',0))>(s.get('cycle',0),s.get('local_step',0)) or s2.get('source')!=s.get('source') or s2.get('state')=='complete'
            if healthy and progressed and s2.get('state')!='failed':
                proof={'first':first,'second':second,'time':now(),'detached':True}
                atomic_json(PROJECT/'runs'/run_id/'detached_verification.json',proof)
                return {'run_id':run_id,'pod_id':r['pod_id'],'state':'full_running','laptop_can_close':True,
                        'forecast_seconds':accepted['forecast_seconds'],'forecast_usd':read_json(PROJECT/'runs'/run_id/'preflight.json')['forecast_usd'],
                        'deadline':r['absolute_deadline']}
        if ((first.get('completion') or {}).get('complete')
                and (first.get('coordinator') or {}).get('phase')=='complete'):
            return {'run_id':run_id,'state':'complete','laptop_can_close':True}
        if first.get('pod_state')=='EXITED':
            cloud.status(run_id)
            live=remote_state(r)
            if live.get('completion.json',{}).get('complete'):
                from ..artifacts.gradient_sources import Snapshot
                snap=live.get('backup_snapshot.json')
                if snap and Snapshot(snap,STATE/'epoch-terminal-check'/run_id).json('completion.json').get('complete'):
                    return {'run_id':run_id,'state':'complete','laptop_can_close':True,'provider_stopped':True}
            raise RuntimeError('Provider stopped before verified full completion; inspect recovered status')
        time.sleep(5)
    raise RuntimeError('Detached full progress not observed before the bounded deadline; inspect status')


def launch(config_path, *, preflight_only=False, resume_id=None):
    from filelock import FileLock
    c=config(config_path)
    with FileLock(str(STATE/'launch-admission.lock'),timeout=10):
        if resume_id:
            prior=cloud.local_resources(resume_id)
            observed=cloud.status(resume_id)  # Settle stopped compute before budget admission.
            if observed['compute_running']:
                raise RuntimeError('Diagnostic compute is already active; inspect status instead of provisioning again')
            live=remote_state(prior)
            # A remotely backed completion proves there is nothing left to execute.
            cloud.reject_completed_cloud_resume(resume_id,prior['repos'])
        checked=preflight(c,resume_id)
        if preflight_only:
            return checked
        archive,code_hash=cloud.source_archive();api=Runpod();key=cloud.register_key(api)
        previous=cloud.local_resources(resume_id) if resume_id else None
        if previous and previous['code_sha256']!=code_hash:
            raise RuntimeError('Resume runtime changed; historical acceptance cannot certify it')
        run_id=resume_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-epoch-ablation'
        launch_no=len([x for x in cloud.ledger()['launches'] if x['run_id']==run_id])+1
        local=PROJECT/'runs'/run_id;local.mkdir(parents=True,exist_ok=True)
        atomic_json(local/'preflight.json',checked)
        source_cache=Path(checked['input_cache']); cache_archive=input_archive(source_cache)
        volume=backup_volume=pod=ssh=resources=None
        try:
            latest=api.request('GET','/catalog/gpus/'+quote(c['required_gpu'],safe='')+'?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=13.2')
            if (latest['price']['secure']!=checked['hourly_usd'] or not any(d['id']==checked['data_center'] and d.get('availability') not in ('NONE','UNAVAILABLE') for d in latest.get('dataCenters',[]))):
                raise RuntimeError('Matching placement/price changed before provisioning; no compute created')
            start=now()
            if previous:
                volume=api.request('GET','/network-volumes/'+previous['volume_id'])
                backup_volume=api.request('GET','/network-volumes/'+previous['backup_volume_id'])
            else:
                volume=create_volume(api,local,run_id,'iib-'+run_id,30,checked['data_center'],'working')
                backup_volume=create_volume(api,local,run_id,'iib-'+run_id+'-backup',10,checked['data_center'],'backup')
            for v,size in ((volume,30),(backup_volume,10)):
                actual=api.request('GET','/network-volumes/'+v['id'])
                if actual['size']!=size or actual.get('dataCenterId',actual.get('dataCenter'))!=checked['data_center']:
                    raise RuntimeError('New volume read-back differs')
            store={'backend':'runpod_s3','bucket':backup_volume['id'],'region':checked['data_center'],'prefix':run_id}
            from .s3_artifacts import probe,check_capacity
            probe(store)
            if previous:
                connection=__import__('model_collapse.cloud.s3_artifacts',fromlist=['client']).client(store)
                stored=sum(item['Size'] for page in connection.get_paginator('list_objects_v2').paginate(Bucket=store['bucket']) for item in page.get('Contents',[]))
                check_capacity(store,10,max(0,6_600_000_000-stored))
            else:
                check_capacity(store,10,6_600_000_000)
            c={**c,'backup_store':store};atomic_json(local/'config.json',c)
            payload={'name':session_name(run_id,launch_no),'image':c['required_image'],'disk':20,'cloud':'SECURE',
                     'gpu':{'id':c['required_gpu'],'count':1,'minCudaVersion':'13.2'},'dataCenterIds':[checked['data_center']],
                     'startSsh':True,'startJupyter':False,'ports':['22/tcp'],
                     'env':{'PUBLIC_KEY':key.with_suffix('.pub').read_text().strip()},
                     'mounts':{'network':[{'path':'/workspace','volumeId':volume['id']}]}}
            atomic_json(local/'pending_pod_request.json',payload);created=now()
            if any(p.get('name')==payload['name'] for p in api.request('GET','/pods')['pods']):
                raise RuntimeError('A Pod with this exact launch name already exists')
            try:
                pod=api.request('POST','/pods',payload)
            except Exception:
                # Reconcile an ambiguous provider response; never repeat the create.
                for _ in range(12):
                    matches=[p for p in api.request('GET','/pods')['pods'] if p.get('name')==payload['name']]
                    if len(matches)==1:
                        pod=matches[0];break
                    time.sleep(5)
                if pod is None:
                    raise
            resources={'run_id':run_id,'launch_id':run_id+f'-{launch_no:02d}','workflow':'epoch_ablation',
                'allocation':ALLOCATION,'kind':'diagnostic','pod_id':pod['id'],'volume_id':volume['id'],
                'backup_volume_id':backup_volume['id'],'backup_store':store,'created_at':created,'volume_created_at':start,
                'compute_started_at':created,'compute_hourly_usd':(checked['hourly_usd']+20*.10/730)*1.15,
                'volume_hourly_usd':30*.07/730,'guard_hourly_usd':checked['guard_hourly_usd'],
                'session_allowance_usd':checked['remaining_task_usd']-checked['retained_seven_days_usd'],
                'absolute_deadline':created+checked['remaining_seconds']-60,
                'acceptance_deadline':validation_deadline(created,checked),
                'data_center':checked['data_center'],'gpu':c['required_gpu'],'image':c['required_image'],
                'code_sha256':code_hash,'runtime_sha256':code_hash,'configuration_sha256':digest(c),
                'starting_balance_usd':checked['account_balance_usd'],'repos':checked['repos'],
                'network_volume_gb':30,'container_disk_gb':20,'remote_root':'/workspace/runs/'+run_id}
            cloud.register_resources(resources)
            # Record local ownership/cost immediately; defer fallible Hub control
            # publication until the independent watchdog and supervisor are alive.
            cloud.update_ledger({**resources,'repos':None},state='provisioning')
            while now()<min(created+360,resources['acceptance_deadline']-90):
                p=api.pod(pod['id'])
                if p['status']=='RUNNING' and p.get('ssh',{}).get('direct'):
                    try:ssh=cloud.ssh_connect(p,key);break
                    except OSError:pass
                time.sleep(5)
            if ssh is None:raise RuntimeError('Pod bootstrap exceeded bounded acceptance time')
            if p['gpu']['id']!=c['required_gpu'] or p['dataCenterId']!=checked['data_center'] or p['image']!=c['required_image']:
                raise RuntimeError('Pod hardware/image read-back differs')
            if p.get('mounts',{}).get('network')!=[{'path':'/workspace','volumeId':volume['id']}]:
                raise RuntimeError('Pod does not mount this launch\'s exact working volume')
            remote=resources['remote_root'];cloud.command(ssh,'mkdir -p /workspace/project '+shlex.quote(remote))
            sftp=ssh.open_sftp();sftp.get_channel().settimeout(90)
            sftp.put(str(archive),'/workspace/epoch-code.tar.gz')
            cloud.command(ssh,'tar -xzf /workspace/epoch-code.tar.gz -C /workspace/project')
            verify="from model_collapse.common import read_json,verify_files; m=read_json('source_bundle.json'); assert m['runtime_sha256']=="+repr(code_hash)+"; verify_files('.',m['files'])"
            cloud.command(ssh,'cd /workspace/project && python3 -c '+shlex.quote(verify))
            if previous:
                archive_controls(sftp,remote,resources['launch_id'])
            cloud.upload_json(sftp,remote+'/resources.json',resources);cloud.upload_json(sftp,remote+'/config.json',c)
            if not checked['previously_accepted']:
                cloud.upload_json(sftp,remote+'/validation_guard_deadline.json',{'deadline':resources['acceptance_deadline']})
            else:
                try:
                    sftp.remove(remote+'/validation_guard_deadline.json')
                except FileNotFoundError:
                    pass
            from huggingface_hub import get_token
            secrets={'RUNPOD_API_KEY':api.key,'HF_TOKEN':os.environ.get('HF_TOKEN') or get_token(),
                     **{n:os.environ.get(n) for n in ('RUNPOD_S3_ACCESS_KEY_ID','RUNPOD_S3_SECRET_ACCESS_KEY')}}
            if not all(secrets.values()):raise RuntimeError('Required transfer credentials are missing')
            with sftp.open('/run/iib-epoch-secrets','w') as f:f.write(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in secrets.items()))
            sftp.chmod('/run/iib-epoch-secrets',0o600)
            prefix='cd /workspace/project && . /run/iib-epoch-secrets && export PYTHONPATH=/workspace/project HF_HOME=/workspace/hf-cache TOKENIZERS_PARALLELISM=false && '
            cloud.command(ssh,prefix+'sh -c '+shlex.quote('setsid python3 -m model_collapse watchdog --root '+remote+f' >/workspace/epoch-watchdog-{launch_no:02d}.log 2>&1 </dev/null &'),timeout=30)
            sftp.put(str(cache_archive),'/workspace/epoch-inputs.tar.gz')
            verify_inputs="import hashlib; assert hashlib.sha256(open('/workspace/epoch-inputs.tar.gz','rb').read()).hexdigest()=="+repr(sha256(cache_archive))
            cloud.command(ssh,'python3 -c '+shlex.quote(verify_inputs))
            cloud.command(ssh,'tar -xzf /workspace/epoch-inputs.tar.gz -C '+shlex.quote(remote))
            cloud.command(ssh,prefix+'python3 -m venv --system-site-packages /opt/iib-venv && /opt/iib-venv/bin/pip install --progress-bar off -e . >/workspace/install.log 2>&1',timeout=360)
            cloud.command(ssh,prefix+'sh -c '+shlex.quote('setsid /opt/iib-venv/bin/python -m model_collapse supervisor --root '+remote+' >'+remote+'/supervisor.log 2>&1 </dev/null &'),timeout=30)
            sftp.close();ssh.close();ssh=None
            cloud.update_ledger(resources,state='running')
        except BaseException:
            if ssh is not None:ssh.close()
            if pod:
                api.stop(pod['id'])
                if resources:
                    stopped,proof=cloud._provider_shutdown(api,api.pod(pod['id']),resources)
                    atomic_json(local/'provider_shutdown.json',proof)
                    validation_cost=0. if checked['previously_accepted'] else max(0,min(stopped,resources['acceptance_deadline'])-resources['created_at'])/3600*resources['guard_hourly_usd']
                    cloud.update_ledger(resources,state='failed',stopped_at=stopped,validation_estimated_usd=validation_cost)
            elif not previous:
                # No compute was created: remove only this launch's recorded empty volumes.
                from filelock import FileLock
                for role in ('working','backup'):
                    owned=read_json(local/('created_'+role+'_volume.json'))
                    if owned:
                        api.request('DELETE','/network-volumes/'+owned['id'])
                        if any(v['id']==owned['id'] for v in api.request('GET','/network-volumes')['networkVolumes']):
                            raise RuntimeError('Task-owned precompute volume deletion was not verified')
                        with FileLock(str(STATE/'ledger.lock'),timeout=30):
                            d=cloud.ledger();d['volumes'][owned['id']]['deleted_at']=now();atomic_json(STATE/'ledger.json',d)
            # Scientific recovery storage is retained once a Pod exists.
            raise
    return verify_detached(run_id)


def cleanup(run_id):
    r=cloud.local_resources(run_id);root=PROJECT/'runs'/run_id
    recovered=read_json(root/'recovery_verified.json',{})
    if recovered.get('metadata_only',True):
        raise RuntimeError('Complete verified checkpoint recovery is required before deleting this diagnostic storage')
    index=read_json(root/'artifact_index.json')
    from ..hub import verify_recovered_files
    verify_recovered_files(root,index['files'])
    api=Runpod();pod=api.pod(r['pod_id'])
    if pod['status']!='EXITED':raise RuntimeError('Diagnostic compute must be provider-stopped before cleanup')
    entries=[x for x in cloud.ledger()['launches'] if x['run_id']==run_id]
    if not any(all(x.get(k)==r[k] for k in ('run_id','launch_id','pod_id','volume_id')) for x in entries):
        raise RuntimeError('Full diagnostic ownership association differs')
    backup_owner=cloud.ledger()['volumes'].get(r['backup_volume_id'])
    if not backup_owner or backup_owner['run_id']!=run_id or backup_owner.get('allocation')!=ALLOCATION:
        raise RuntimeError('Private backup association differs from this task')
    cloud.cleanup(run_id)
    api.request('DELETE','/network-volumes/'+r['backup_volume_id'])
    if any(v['id']==r['backup_volume_id'] for v in api.request('GET','/network-volumes')['networkVolumes']):
        raise RuntimeError('Private backup volume deletion not verified')
    from filelock import FileLock
    with FileLock(str(STATE/'ledger.lock'),timeout=30):
        d=cloud.ledger();d['volumes'][r['backup_volume_id']]['deleted_at']=now();atomic_json(STATE/'ledger.json',d)
    return {'run_id':run_id,'compute_stopped':True,'working_and_backup_volumes_removed':True}
