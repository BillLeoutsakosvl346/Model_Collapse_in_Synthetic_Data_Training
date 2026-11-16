import json
from types import SimpleNamespace

import pytest

from model_collapse.common import atomic_json, read_json
from model_collapse.common import digest
from model_collapse.cloud import epoch_ablation as cloud
from model_collapse.workflow.epoch_ablation import validate_local


def test_volume_create_timeout_reconciles_one_owned_resource(tmp_path,monkeypatch):
    calls=[];recorded=[]
    def request(method,path,body=None):
        calls.append((method,path))
        if method=='POST':raise TimeoutError('ambiguous create')
        return {'networkVolumes':([] if len(calls)==1 else [{'id':'new','name':'unique-task','size':30}])}
    monkeypatch.setattr(cloud,'record_volume',lambda v,run,started,size:recorded.append((v['id'],run,size)))
    result=cloud.create_volume(SimpleNamespace(request=request),tmp_path,'r','unique-task',30,'EU-RO-1','working')
    assert result['id']=='new' and recorded==[('new','r',30)]
    assert sum(m=='POST' for m,p in calls)==1
    assert read_json(tmp_path/'created_working_volume.json')['id']=='new'


def test_no_matching_create_result_never_adopts_other_volume(tmp_path,monkeypatch):
    def request(method,path,body=None):
        if method=='POST':raise TimeoutError('ambiguous create')
        return {'networkVolumes':[{'id':'old','name':'different'}]}
    monkeypatch.setattr(cloud,'record_volume',lambda *a:pytest.fail('Must not adopt another resource'))
    with pytest.raises(TimeoutError):
        cloud.create_volume(SimpleNamespace(request=request),tmp_path,'r','unique-task',30,'EU-RO-1','working')


def test_validate_does_not_trust_declared_nine_branch_totals(tmp_path):
    atomic_json(tmp_path/'completion.json',{'complete':True,'updates':711,'branches':9})
    with pytest.raises(RuntimeError,match='Missing'):
        validate_local(tmp_path)


def test_detached_null_completion_keeps_waiting(tmp_path,monkeypatch):
    monkeypatch.setattr(cloud.cloud,'local_resources',lambda x:{'absolute_deadline':100.})
    monkeypatch.setattr(cloud,'now',lambda:50.)
    monkeypatch.setattr(cloud,'_observe',lambda r:{'completion':None,'status':{'state':'running','phase':'acceptance'}})
    def done(*args):raise RuntimeError('bounded wait observed')
    monkeypatch.setattr(cloud.time,'sleep',done)
    with pytest.raises(RuntimeError,match='bounded wait observed'):
        cloud.verify_detached('r',timeout=10)


def test_remote_state_uses_safe_descriptor_and_explicit_object_keys(monkeypatch):
    import io
    from model_collapse.cloud import s3_artifacts
    from model_collapse.cloud.s3_artifacts import validate_store
    seen=[];cfg={'workflow':'epoch_ablation'}
    def connection(store):
        validate_store(store)
        def get_object(**kw):
            seen.append(kw['Key'])
            name=kw['Key'].split('/')[-1]
            value=cfg if name=='config.json' else {}
            return {'Body':io.BytesIO(json.dumps(value).encode())}
        return SimpleNamespace(get_object=get_object)
    monkeypatch.setattr(s3_artifacts,'client',connection)
    resources={'volume_id':'example','data_center':'EU-RO-1','run_id':'run-01','configuration_sha256':digest(cfg)}
    result=cloud.remote_state(resources)
    assert result['config.json']==cfg
    assert seen==['runs/run-01/'+n for n in ('config.json','completion.json','backup_snapshot.json','artifact_index.json','acceptance.json')]


def test_accepted_resume_uses_total_time_even_when_testing_money_is_exhausted():
    checked={'previously_accepted':True,'remaining_seconds':3000,'remaining_validation_usd':-.01,'guard_hourly_usd':.66}
    assert cloud.validation_deadline(1000,checked)==4000
    assert cloud.session_name('run',1)!=cloud.session_name('run',2)


def test_resume_archives_old_shutdown_requests_without_touching_acceptance():
    existing={'/work/stop_request.json','/work/supervisor.json','/work/acceptance.json'};moves=[]
    def stat(path):
        if path not in existing:raise FileNotFoundError(path)
    def rename(a,b):moves.append((a,b));existing.remove(a);existing.add(b)
    api=SimpleNamespace(mkdir=lambda p:existing.add(p),stat=stat,rename=rename)
    cloud.archive_controls(api,'/work','session-02')
    assert '/work/acceptance.json' in existing
    assert '/work/stop_request.json' not in existing and '/work/supervisor.json' not in existing
    assert len(moves)==2
