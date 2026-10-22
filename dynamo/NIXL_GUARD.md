# Interim NIXL rollout guard

This opt-in guard stops a V1 `separate_async` training step when any P/D worker
publishes a nonzero `vllm:nixl_num_failed_transfers_total`. It runs in the trainer's
sampling loop, including while waiting for replacement trajectories, and again
before returning a batch for training. Missing workers, missing counters, scrape
failures, and changed counter creation times (when exported) also stop sampling. A failure stays
latched for the life of the sampler.

Use a fresh directory shared by the trainer and every rollout node for
`VERL_DYNAMO_WORKER_METRICS_DIR`. Keep worker system metrics and vLLM log statistics
enabled. Configure the **total number of P/D engine processes across all nodes**,
not the number of TP ranks, for example:

```bash
trainer.v1.sampler.custom_sampler.path=/opt/verl/recipe/dynamo/nixl_guarded_sampler.py
trainer.v1.sampler.custom_sampler.name=NixlGuardedReplayBuffer
+trainer.v1.sampler.sampler_kwargs.nixl_guard_expected_workers=2
```

The recipe records P/D targets separately from aggregated hybrid/validation
workers, which do not necessarily expose NIXL counters. This sampler is intended
for the current separate-async gate; it does not add DAPO group filtering.

This is a monitoring safeguard. Publication lag can hide a recent transfer error
until a subsequent poll; the guard cannot certify already-consumed trajectories
or undo an optimizer update. An unobserved reset at the same endpoint is also
undetectable if the exporter omits counter creation timestamps. The engine must still propagate receive failures
before generating tokens. On an alert, stop the rollout step, discard trajectories
from the failure window, restart both workers of the affected P/D pair (restarting
one side alone did not recover in testing), and verify healthy transfers
and output correctness before resuming from a clean state.
