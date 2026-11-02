"""Standalone launches obey explicit task authorization without changing the reserve."""
from pathlib import Path

import pytest

from model_collapse import remote
from model_collapse.common import PROJECT


def test_explicit_ledger_ceiling_overrides_original_default():
    assert remote.task_budget({"total_budget_usd": 50}) == 50
    assert remote.task_budget({"total_budget_usd": 10}) == 10
    assert remote.task_budget({"launches": []}) == remote.TOTAL_BUDGET


@pytest.mark.parametrize("amount", [True, None, 0, -1, float("nan"), float("inf")])
def test_invalid_explicit_authorization_never_falls_back(amount):
    with pytest.raises(RuntimeError, match="explicit authorized cumulative budget"):
        remote.task_budget({"total_budget_usd": amount})


@pytest.mark.parametrize("ceiling,admitted", [(50, True), (25, False)])
def test_standalone_admission_uses_recorded_ceiling_before_any_mutation(tmp_path, monkeypatch, ceiling, admitted):
    from model_collapse.workflow import initial_data

    class AdmissionReached(Exception):
        pass

    class API:
        def request(self, method, path, body=None):
            assert method == "GET" and path == "/pods", "No cloud mutation is allowed in this test"
            return {"pods": []}

        def candidates(self):
            return [{"gpu": "NVIDIA RTX PRO 4000 Blackwell", "data_center": "EU-RO-1", "hourly_usd": .57}]

    monkeypatch.setattr(remote, "STATE", tmp_path / ".state")
    monkeypatch.setattr(remote, "load_secrets", lambda: None)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "preflight", lambda **kwargs: {"account": {"clientBalance": 20}})
    monkeypatch.setattr(remote, "ledger", lambda: {"total_budget_usd": ceiling, "launches": []})
    monkeypatch.setattr(remote, "accrued", lambda d, kind=None: .1 if kind == "smoke" else 39.6)
    monkeypatch.setattr(remote, "campaign_accrued", lambda d, campaign: .1)
    monkeypatch.setattr(remote, "source_archive", lambda: (Path("unused"), "runtime"))
    monkeypatch.setattr(initial_data, "resolve_initial_dataset", lambda c: c)

    def stop_before_provisioning(c):
        raise AdmissionReached("Admission passed with the authorized ceiling and unchanged reserve")

    monkeypatch.setattr(remote, "verify_start_checkpoint", stop_before_provisioning)
    expected = AdmissionReached if admitted else RuntimeError
    message = "Admission passed" if admitted else "Insufficient budget"
    with pytest.raises(expected, match=message):
        remote.launch(smoke=True, config_path=PROJECT / "experiments/opt125m_wikitext2/continued_real_only_final5/smoke.yaml")
    assert remote.RESERVE == 2


def test_preflight_reports_effective_authorization(tmp_path, monkeypatch):
    class API:
        def account(self):
            return {"isAutoPayEnabled": False, "enableBackupPayments": False}

        def candidates(self):
            return []

    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "load_secrets", lambda: None)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "destinations", lambda **kwargs: {})
    monkeypatch.setattr(remote, "ledger", lambda: {"total_budget_usd": 50, "launches": []})
    monkeypatch.setattr(remote, "accrued", lambda d: 39.6)
    assert remote.preflight()["total_task_budget_usd"] == 50
