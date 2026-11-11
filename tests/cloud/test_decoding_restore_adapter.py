import copy
import pytest
import yaml
from pathlib import Path

from scripts.operations.finish_decoding_validation import require_restoration_proof
from model_collapse.workflow.completion import acceptance_checks
from model_collapse.workflow.profiles import protocol_fingerprint


def test_restoration_resume_requires_every_scientific_check_and_executed_source():
    c = yaml.safe_load(Path("comparisons/fineweb_completion64_decoding/smoke.yaml").read_text())
    proof = {"source_code_sha256": "executed", "protocol_fingerprint": protocol_fingerprint(c),
             "checks": {k: True for k in acceptance_checks(c)}}
    require_restoration_proof(proof, c, "executed")
    wrong = copy.deepcopy(proof)
    wrong["checks"]["sampling_recovery_byte_identity"] = False
    with pytest.raises(RuntimeError):
        require_restoration_proof(wrong, c, "executed")
    with pytest.raises(RuntimeError):
        require_restoration_proof(proof, c, "different_source")
