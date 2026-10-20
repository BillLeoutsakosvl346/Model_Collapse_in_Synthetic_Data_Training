import math

import numpy as np
import pytest
import torch

from model_collapse.common import config
from model_collapse.data import pack_rows
from model_collapse.engine import diversity, evaluate, shifted_nll


def test_shifted_nll_uses_next_tokens_and_counts_targets():
    logits = torch.tensor([[[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]]])
    ids = torch.tensor([[0, 2, 1]])
    scores = shifted_nll(logits, ids)
    expected = [-math.log(math.exp(3) / sum(math.exp(x) for x in [1, 2, 3]))] * 2
    assert scores.shape == (1, 2)
    assert scores.tolist()[0] == pytest.approx(expected)


def test_eval_corpus_ppl_is_invariant_to_batch_size(monkeypatch):
    import model_collapse.engine as engine
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    class Model:
        def eval(self):
            return self
        def __call__(self, input_ids, **kwargs):
            class Result:
                pass
            r = Result()
            r.logits = torch.nn.functional.one_hot(input_ids, 3).float() * 2
            return r
    values = np.array([[0, 1, 2], [1, 1, 1], [2, 2, 0]], dtype=np.int32)
    a = evaluate(Model(), values, {"eval_batch": 1, "precision": "fp32"})
    b = evaluate(Model(), values, {"eval_batch": 2, "precision": "fp32"})
    assert a["scored_tokens"] == 6
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-12)
    assert a["perplexity"] == pytest.approx(math.exp(a["nll_sum"] / 6))
    assert a["perplexity"] != pytest.approx(a["mean_sequence_perplexity"])


def test_packing_preserves_rows_and_only_drops_split_remainder():
    def tok(rows, **kwargs):
        return {"input_ids": [[99] + list(map(int, r.split())) for r in rows]}
    blocks, dropped = pack_rows(["1 2", "3 4", "5"], tok, block_size=4)
    assert blocks.tolist() == [[99, 1, 2, 99], [3, 4, 99, 5]]
    assert dropped == 0
    blocks, dropped = pack_rows(["1 2", "3"], tok, block_size=4)
    assert blocks.tolist() == [[99, 1, 2, 99]]
    assert dropped == 1


def test_diversity_definitions_and_no_cross_sequence_ngrams():
    d = diversity(np.array([[1, 2, 1, 2], [1, 2, 1, 2]]))
    assert d["distinct_1"] == 2 / 8
    assert d["distinct_2"] == 2 / 6
    assert d["duplicate_continuation_rate"] == .5
    assert d["repeated_4gram_rate"] == .5
    assert d["repeated_4gram_denominator"] == 2


def test_configs_lock_scientific_settings():
    from model_collapse.common import PROJECT
    full = config(PROJECT / "configs/full.yaml")
    smoke = config(PROJECT / "configs/smoke.yaml")
    assert full["train_blocks"] is None and full["epochs"] == 5
    assert smoke["train_blocks"] == 256 and smoke["generations"] == 2
    for name in ("model_revision", "dataset_revision", "block_size", "num_beams", "effective_batch", "repetition_penalty"):
        assert full[name] == smoke[name]
