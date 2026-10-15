"""The sampler must stop training, including while no successful samples arrive."""

from io import BytesIO
from unittest.mock import Mock

import pytest
from recipe.dynamo import nixl_failure_guard as module
from recipe.dynamo.nixl_failure_guard import NixlFailureGuard


@pytest.fixture
def monitored(tmp_path, monkeypatch):
    (tmp_path / "r0_n0.pd-endpoints").write_text("host-a:10001\nhost-b:10002\n")
    bodies = {
        f"http://host-{host}:{port}/metrics": 'vllm:nixl_num_failed_transfers_total{engine="0"} 0\n'
        'vllm:nixl_num_failed_transfers_created{engine="0"} 1234\n'
        for host, port in [("a", 10001), ("b", 10002)]
    }

    def read(url, timeout):
        value = bodies[url]
        if isinstance(value, Exception):
            raise value
        return BytesIO(value.encode())

    monkeypatch.setattr(module, "urlopen", read)
    return NixlFailureGuard(tmp_path, 2), bodies, tmp_path


def test_healthy_workers_allow_sampling(monitored):
    guard, _, _ = monitored
    guard.check()
    guard.check()


def test_failure_on_any_worker_is_latched_even_after_restart(monitored):
    guard, bodies, _ = monitored
    guard.check()
    url = "http://host-b:10002/metrics"
    original = bodies[url]
    bodies[url] = original.replace("} 0", "} 1")
    with pytest.raises(RuntimeError, match="host-b"):
        guard.check()
    bodies[url] = original
    with pytest.raises(RuntimeError, match="Discard trajectories"):
        guard.check()


@pytest.mark.parametrize("fault", ["missing_counter", "scrape_error", "counter_reset", "nan", "missing_worker"])
def test_lost_or_replaced_telemetry_fails_closed(monitored, fault):
    guard, bodies, root = monitored
    guard.check()
    url = "http://host-b:10002/metrics"
    if fault == "missing_counter":
        bodies[url] = "vllm:num_requests_running 3\n"
    elif fault == "scrape_error":
        bodies[url] = TimeoutError("worker unreachable")
    elif fault == "counter_reset":
        bodies[url] = bodies[url].replace("1234", "1235")
    elif fault == "nan":
        bodies[url] = bodies[url].replace("} 0", "} NaN")
    else:
        (root / "r0_n0.pd-endpoints").write_text("host-a:10001\n")
    with pytest.raises(RuntimeError, match="guard stopped sampling"):
        guard.check()


def test_initial_nonzero_counter_is_rejected(monitored):
    guard, bodies, _ = monitored
    url = "http://host-a:10001/metrics"
    bodies[url] = bodies[url].replace("} 0", "} 9")
    with pytest.raises(RuntimeError, match="transfer failure"):
        guard.check()


def test_guard_stops_waiting_sampler_and_prevents_batch_return(monkeypatch):
    from recipe.dynamo.nixl_guarded_sampler import NixlGuardedReplayBuffer, ReplayBufferAsync

    sampler = object.__new__(NixlGuardedReplayBuffer)
    sampler.nixl_guard = Mock()
    sampler.nixl_guard.check.side_effect = RuntimeError("injected NIXL failure")
    sync = Mock()
    materialize = Mock(return_value="would train on this batch")
    monkeypatch.setattr(ReplayBufferAsync, "_sync_metadata_from_transfer_queue", sync)
    monkeypatch.setattr(ReplayBufferAsync, "_materialize_batch", materialize)
    with pytest.raises(RuntimeError, match="injected NIXL failure"):
        sampler._sync_metadata_from_transfer_queue()
    sync.assert_not_called()
    with pytest.raises(RuntimeError, match="injected NIXL failure"):
        sampler._materialize_batch("train", [], {})


def test_endpoint_discovery_includes_all_pd_workers_and_excludes_aggregated(tmp_path, monkeypatch):
    from recipe.dynamo.dynamo_async_server import DynamoHttpServer

    monkeypatch.setenv("VERL_DYNAMO_WORKER_METRICS_DIR", str(tmp_path))
    server = object.__new__(DynamoHttpServer)
    server.replica_rank = 2
    server.node_rank = 1
    server._record_worker_metrics_endpoint("p:10001", role="prefill")
    server._record_worker_metrics_endpoint("d:10002", role="decode")
    server._record_worker_metrics_endpoint("a:10003", role="agg")
    assert (tmp_path / "r2_n1.pd-endpoints").read_text().splitlines() == ["p:10001", "d:10002"]
    assert (tmp_path / "r2_n1.endpoints").read_text().splitlines() == ["p:10001", "d:10002", "a:10003"]
