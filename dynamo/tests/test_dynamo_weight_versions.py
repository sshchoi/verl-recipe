import asyncio
import json
from types import SimpleNamespace

import pytest

from recipe.dynamo.dynamo_async_server import DynamoHttpServer
from recipe.dynamo.dynamo_worker_extension import vLLMDynamoColocateWorkerExtension
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


def state(rank, version=4, updates=2):
    return {"local_rank": rank, "version": version, "completed_updates": updates}


def server_with_states(states):
    server = object.__new__(DynamoHttpServer)
    server.config = SimpleNamespace(tensor_model_parallel_size=2, engine_kwargs={})
    server._control_endpoints = ["prefill", "decode"]
    server.replica_rank = server.node_rank = 0
    server.global_steps = 3

    async def collective_rpc(method, timeout):
        assert method == "get_weight_update_state"
        assert timeout == 8
        return states

    server.collective_rpc = collective_rpc
    return server


def test_all_tp_workers_acknowledge_before_coordinator_advances(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_DYNAMO_LOG_DIR", str(tmp_path))
    states = [[state(0), state(1)], [state(0), state(1)]]
    server = server_with_states(states)
    asyncio.run(server.set_global_steps(4))
    assert server.global_steps == 4
    record = json.loads((tmp_path / "weight_versions_replica0_node0.jsonl").read_text())
    assert record["workers"] == states


@pytest.mark.parametrize("states", [
    [],
    [[state(0), state(1)]],
    [[state(0), state(1)], []],
    [[state(0), state(1)], [state(0), state(1, version=3)]],
    [[state(0), state(1)], [state(0), state(1, updates=0)]],
    [[state(0), state(1)], [state(0), state(0)]],
    [[state(0), state(1)], [state(0), None]],
])
def test_missing_or_stale_worker_prevents_version_advance(states):
    server = server_with_states(states)
    with pytest.raises(RuntimeError, match="unconfirmed on every TP worker"):
        asyncio.run(server.set_global_steps(4))
    assert server.global_steps == 3


def test_worker_records_version_only_after_successful_reload(monkeypatch):
    worker = object.__new__(vLLMDynamoColocateWorkerExtension)
    worker.local_rank = 1

    def success(self, **kwargs):
        assert "weight_version" not in kwargs
        return "loaded"

    monkeypatch.setattr(vLLMColocateWorkerExtension, "update_weights_from_ipc", success)
    assert worker.update_weights_from_ipc(weight_version=4) == "loaded"
    assert worker.get_weight_update_state() == state(1, version=4, updates=1)

    def failure(self, **kwargs):
        raise RuntimeError("post-load failed")

    monkeypatch.setattr(vLLMColocateWorkerExtension, "update_weights_from_ipc", failure)
    with pytest.raises(RuntimeError, match="post-load failed"):
        worker.update_weights_from_ipc(weight_version=5)
    assert worker.get_weight_update_state() == state(1, version=4, updates=1)
