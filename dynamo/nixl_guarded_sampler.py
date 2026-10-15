"""Opt-in NIXL counter guard at the V1 separate_async training boundary."""

import os

from recipe.dynamo.nixl_failure_guard import NixlFailureGuard

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync


class NixlGuardedReplayBuffer(ReplayBufferAsync):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.trainer_mode != "separate_async":
            raise ValueError("NixlGuardedReplayBuffer supports V1 separate_async only")
        options = self.sampler_kwargs or {}
        endpoint_dir = options.get("nixl_guard_endpoint_dir") or os.environ.get("VERL_DYNAMO_WORKER_METRICS_DIR")
        if not endpoint_dir:
            raise ValueError("NIXL guard requires a fresh, shared worker metrics directory")
        self.nixl_guard = NixlFailureGuard(
            endpoint_dir, options["nixl_guard_expected_workers"], options.get("nixl_guard_timeout_s", 5.0)
        )

    def _sync_metadata_from_transfer_queue(self):
        # This runs on every sampling poll, including when all trajectories fail.
        # Raising in a background agent task alone would just drop that prompt.
        self.nixl_guard.check()
        return super()._sync_metadata_from_transfer_queue()

    def _materialize_batch(self, *args, **kwargs):
        batch = super()._materialize_batch(*args, **kwargs)
        self.nixl_guard.check()
        return batch
