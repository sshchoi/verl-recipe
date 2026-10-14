import asyncio
import pickle
from types import SimpleNamespace

import pytest
import zmq
import zmq.asyncio

from recipe.dynamo.dynamo_async_server import DynamoHttpServer, DynamoReplica, RolloutMode


def replica(mode, *, standalone_only=False, sleeping=True, roles=None, enabled=True):
    result = object.__new__(DynamoReplica)
    result.rollout_mode = mode
    result.config = SimpleNamespace(
        free_cache_engine=sleeping,
        engine_kwargs={"dynamo": {"disaggregation": {
            "enabled": enabled,
            "standalone_only": standalone_only,
            "shard_roles": roles if roles is not None else ["prefill", "decode"],
        }}},
    )
    return result


@pytest.mark.parametrize("mode", [mode for mode in RolloutMode if mode != RolloutMode.STANDALONE])
@pytest.mark.parametrize("standalone_only", [False, True])
@pytest.mark.parametrize("sleeping", [False, True])
def test_pd_never_applies_to_hybrid_or_validation(mode, standalone_only, sleeping):
    assert replica(mode, standalone_only=standalone_only, sleeping=sleeping)._pd_shard_roles() == []


def test_standalone_pd_roles():
    assert replica(RolloutMode.STANDALONE, sleeping=False)._pd_shard_roles() == ["prefill", "decode"]


@pytest.mark.parametrize("roles", [["decode"], ["prefill"], ["prefill", "decode", "agg"]])
def test_pd_requires_both_valid_roles(roles):
    with pytest.raises(ValueError, match="both prefill and decode"):
        replica(RolloutMode.STANDALONE, roles=roles)._pd_shard_roles()


@pytest.mark.parametrize("result", [True, False])
def test_cache_reset_checks_engine_boolean(result):
    async def run():
        context = zmq.asyncio.Context.instance()
        responder = context.socket(zmq.REP)
        port = responder.bind_to_random_port("tcp://127.0.0.1")
        server = object.__new__(DynamoHttpServer)
        server._control_endpoints = [f"tcp://127.0.0.1:{port}"]

        async def respond():
            request = pickle.loads(await responder.recv())
            assert request["method"] == "reset_prefix_cache"
            await responder.send(pickle.dumps({"ok": True, "result": result}))

        task = asyncio.create_task(respond())
        try:
            if result:
                await server._engine_method_all("reset_prefix_cache", timeout=2)
            else:
                with pytest.raises(RuntimeError, match="refusing stale-policy"):
                    await server._engine_method_all("reset_prefix_cache", timeout=2)
            await task
        finally:
            task.cancel()
            responder.close(0)

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["reset_timeout", "reset_failure", "version_timeout"])
def test_update_fails_closed_on_unconfirmed_cache_or_version(monkeypatch, failure):
    from unittest.mock import AsyncMock

    from recipe.dynamo.dynamo_vllm_rollout import VllmDynamoServerAdapter
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    async def run():
        adapter = object.__new__(VllmDynamoServerAdapter)
        adapter.rollout_rank = 0
        adapter.use_shm = False
        adapter.zmq_handle = "unused"
        adapter.config = SimpleNamespace(
            free_cache_engine=False,
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1),
        )
        adapter._is_node_control_rank = lambda: True
        clear = AsyncMock(side_effect=RuntimeError("reset failed") if failure == "reset_failure" else None)
        version = AsyncMock()
        adapter.server_handle = SimpleNamespace(
            clear_kv_cache=SimpleNamespace(remote=clear),
            set_global_steps=SimpleNamespace(remote=version),
        )
        adapter._execute_method = AsyncMock(return_value=None)
        monkeypatch.setattr(BucketedWeightSender, "__init__", lambda self, **kwargs: None)
        monkeypatch.setattr(BucketedWeightSender, "async_send_weights", AsyncMock())
        original_wait_for = asyncio.wait_for

        async def wait_for(awaitable, timeout):
            if (failure == "reset_timeout" and timeout == 30) or (failure == "version_timeout" and timeout == 10):
                awaitable.close()
                raise asyncio.TimeoutError
            return await original_wait_for(awaitable, timeout)

        monkeypatch.setattr(asyncio, "wait_for", wait_for)
        message = {"reset_timeout": "refusing stale-policy", "reset_failure": "reset failed",
                   "version_timeout": "weight version is unconfirmed"}[failure]
        with pytest.raises(RuntimeError, match=message):
            await adapter.update_weights(iter(()), global_steps=4)
        if failure != "version_timeout":
            version.assert_not_called()

    asyncio.run(run())
