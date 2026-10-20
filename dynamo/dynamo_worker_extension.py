# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM worker_extension_cls for the dynamo backend.

Each DP shard in the dynamo topology is a separate ``dynamo.vllm`` subprocess,
so two shards' TP rank 0 would both compute ``self.local_rank == 0`` and
connect to the same IPC socket file. DynamoHttpServer disambiguates by
injecting ``VERL_ZMQ_BASE_TRAINER_RANK=<shard rank offset>`` per subprocess —
verl's base ``vLLMColocateWorkerExtension._get_zmq_handle`` consumes it
natively (``int(base) + dp-resolved local rank``; dynamo shards run dp=1 so
the resolver is the identity), producing the node-global socket rank the
trainer/CE sender side computes. No ``_get_zmq_handle`` override needed.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


@contextmanager
def _bucketed_tied_weights(model):
    """Validate vLLM 0.31 tied embeddings over an entire IPC reload.

    AutoWeightsLoader checks a complete checkpoint per call, whereas veRL
    invokes model.load_weights once per bucket. Filter redundant aliases per
    bucket, but still require their canonical parameters to load in this sync.
    """
    from vllm.model_executor.models import utils

    get_aliases = getattr(utils, "_get_tied_embedding_params", None)
    aliases = get_aliases(model) if get_aliases else {}
    if not aliases:
        yield
        return
    original = model.load_weights
    had_override = "load_weights" in model.__dict__
    loaded, skipped = set(), set()

    def load_bucket(weights):
        def canonical_weights():
            for name, tensor in weights:
                if name in aliases:
                    skipped.add(name)
                else:
                    yield name, tensor

        result = original(canonical_weights())
        loaded.update(result or ())
        return result

    model.load_weights = load_bucket
    try:
        yield
        missing = {aliases[name] for name in skipped} - loaded
        if missing:
            raise RuntimeError(f"Tied weights missing from completed IPC reload: {sorted(missing)}")
    finally:
        if had_override:
            model.load_weights = original
        else:
            del model.load_weights


class vLLMDynamoColocateWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM worker mixin for verl × dynamo."""

    def update_weights_from_ipc(self, *args, weight_version=None, **kwargs):
        """Run verl's weight reload inside vLLM's config context.

        vLLM 0.20's FlashInfer MoE post-load path calls
        get_current_vllm_config(). Native vLLM sets that context around engine
        internals, but verl invokes this worker extension through
        collective_rpc, so set it explicitly in the TP worker process.
        """
        vllm_config = getattr(getattr(self, "model_runner", None), "vllm_config", None)
        with ExitStack() as stack:
            if vllm_config is not None:
                from vllm.config import set_current_vllm_config

                stack.enter_context(set_current_vllm_config(vllm_config))
                for model in self._iter_all_models():
                    stack.enter_context(_bucketed_tied_weights(model))
            result = super().update_weights_from_ipc(*args, **kwargs)
        # Acknowledge only after IPC and all post-load processing succeeded.
        self._verl_weight_version = weight_version
        self._verl_weight_updates = getattr(self, "_verl_weight_updates", 0) + 1
        return result

    def get_weight_update_state(self):
        """Read back this TP worker's completed reload, not a coordinator stamp."""
        return {
            "version": getattr(self, "_verl_weight_version", None),
            "completed_updates": getattr(self, "_verl_weight_updates", 0),
            "local_rank": self.local_rank,
        }
