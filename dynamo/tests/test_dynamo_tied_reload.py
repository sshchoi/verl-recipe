"""Exercise the installed vLLM loader with tied tensors split across IPC buckets."""

import pytest
import torch
from recipe.dynamo.dynamo_worker_extension import _bucketed_tied_weights
from torch import nn
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import utils as vllm_utils

# Older vLLM (e.g. 0.28) has no tied-embedding load check, so the wrapper is a no-op there.
pytestmark = pytest.mark.skipif(
    not hasattr(vllm_utils, "_get_tied_embedding_params"),
    reason="installed vLLM has no tied-embedding load check",
)


class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        # Only loading is exercised; no GPU kernels or distributed construction.
        embed = object.__new__(VocabParallelEmbedding)
        nn.Module.__init__(embed)
        embed.weight = nn.Parameter(torch.zeros(3, 2))
        self.model.embed_tokens = embed
        self.lm_head = embed

    def load_weights(self, weights):
        return vllm_utils.AutoWeightsLoader(self).load_weights(weights)


@pytest.mark.parametrize("alias_first", [False, True])
def test_tied_alias_in_different_bucket(alias_first):
    model = TiedModel()
    weights = [("model.embed_tokens.weight", torch.ones(3, 2)), ("lm_head.weight", torch.full((3, 2), 9.0))]
    if alias_first:
        weights.reverse()
    with _bucketed_tied_weights(model):
        for weight in weights:
            model.load_weights([weight])
    assert torch.equal(model.model.embed_tokens.weight, torch.ones(3, 2))
    assert "load_weights" not in model.__dict__
    # The ordinary complete-checkpoint validation remains installed afterward.
    with pytest.raises(ValueError, match="was skipped because it is tied"):
        model.load_weights([("lm_head.weight", torch.ones(3, 2))])


def test_missing_canonical_is_still_rejected_at_end_of_sync():
    model = TiedModel()
    with pytest.raises(RuntimeError, match="Tied weights missing"):
        with _bucketed_tied_weights(model):
            model.load_weights([("lm_head.weight", torch.ones(3, 2))])
    assert "load_weights" not in model.__dict__


def test_original_failure_is_preserved_and_loader_restored():
    model = TiedModel()
    with pytest.raises(ValueError, match="injected"):
        with _bucketed_tied_weights(model):
            raise ValueError("injected")
    assert "load_weights" not in model.__dict__
