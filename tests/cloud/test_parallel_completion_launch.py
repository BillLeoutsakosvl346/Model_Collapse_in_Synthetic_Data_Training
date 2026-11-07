import copy

import pytest

from model_collapse.common import config
from model_collapse.workflow.profiles import protocol_fingerprint
from scripts.operations.launch_parallel_completion_mixtures import check_parallel_full_pods


def setup_scope(monkeypatch):
    from model_collapse import remote
    c = config("comparisons/fineweb_completion64_mixtures/full.yaml")
    owner = {"run_id": "original", "launch_id": "launch", "pod_id": "protected", "volume_id": "volume",
             "kind": "full", "workflow": "completion_comparison"}
    scope = {"protected_run_id": "original", "protected_owner": owner,
             "protocol_fingerprint": protocol_fingerprint(c)}
    state = {"allocations": {"fineweb_completion64_mixtures": {"parallel_full_scope": scope}}}
    pod = {"id": "protected", "name": "iib-original", "status": "RUNNING",
           "mounts": {"network": [{"volumeId": "volume"}]}}
    monkeypatch.setattr(remote, "local_resources", lambda name: owner)
    return c, state, pod


def test_scoped_parallel_full_admission_allows_only_the_original_owner(monkeypatch):
    c, state, pod = setup_scope(monkeypatch)
    check_parallel_full_pods(c, [pod], state)
    check_parallel_full_pods(c, [], state)
    for other in ({**pod, "id": "unknown"}, {**pod, "mounts": {}}, {**pod, "name": "different"}):
        with pytest.raises(RuntimeError):
            check_parallel_full_pods(c, [other], state)
    with pytest.raises(RuntimeError):
        check_parallel_full_pods(c, [pod, {**pod, "id": "second"}], state)


def test_parallel_launch_rejects_changed_protocol_and_authorization(monkeypatch):
    c, state, pod = setup_scope(monkeypatch)
    with pytest.raises(RuntimeError):
        check_parallel_full_pods({**c, "learning_rate": 6e-6}, [pod], state)
    with pytest.raises(RuntimeError):
        check_parallel_full_pods(c, [pod], {})
    changed = copy.deepcopy(state)
    changed["allocations"]["fineweb_completion64_mixtures"]["parallel_full_scope"]["protected_owner"] = {
        "run_id": "original", "launch_id": "different", "pod_id": "protected", "volume_id": "volume"}
    with pytest.raises(RuntimeError):
        check_parallel_full_pods(c, [pod], changed)
