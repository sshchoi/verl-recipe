#!/usr/bin/env bash
set -xeuo pipefail

# Two-GPU prefill/decode (P/D) smoke for the Dynamo rollout path.
#
# Runs V0 sync GRPO with the shared pool split into one prefill and one decode
# vLLM shard (engine_kwargs.dynamo.disaggregation); NIXL moves the KV cache.
# Engines stay resident: vLLM does not re-register NIXL KV memory after
# sleep/wake, so P/D with free_cache_engine=True is not covered here.
#
# Passes only if both roles launched, the frontend activated its prefill router
# (otherwise requests silently run decode-only), and the rollout-vs-actor
# logprob pearson reaches MIN_PEARSON. With the default data every reward is
# equal, so the gradient is zero: this checks wiring, not policy drift.
#
# Requires: verl >= REQUIRED_VERL.txt pin, vLLM with NIXL, 2 GPUs.
# 4 prompts x n=2 = 8 samples, so V0's 8 agent-loop workers get equal chunks.
# Env: MODEL_PATH / TRAIN_FILE / TEST_FILE / RAY_DATA_HOME; MIN_PEARSON (0.99); SMOKE_LOG.

project_name=${PROJECT_NAME:-verl-dynamo}
exp_name=${EXP_NAME:-dynamo-pd-smoke}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

TOTAL_STEPS=${TOTAL_STEPS:-2}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
MIN_PEARSON=${MIN_PEARSON:-0.99}
SMOKE_LOG=${SMOKE_LOG:-"${TMPDIR:-/tmp}/smoke_dynamo_pd_$$.log"}
export VERL_DYNAMO_LOG_DIR=${VERL_DYNAMO_LOG_DIR:-"${TMPDIR:-/tmp}/smoke_dynamo_pd_$$"}

export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register}"
DYNAMO_CONFIG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../config" && pwd)

python3 -m verl.trainer.main_ppo \
    --config-path "${DYNAMO_CONFIG_DIR}" \
    --config-name=dynamo_trainer \
    "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES}'" \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=4 \
    data.val_batch_size=1 \
    data.train_max_samples=4 \
    data.val_max_samples=1 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=false \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=round-robin \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.disaggregation.enabled=true \
    "++actor_rollout_ref.rollout.engine_kwargs.dynamo.disaggregation.shard_roles=[prefill,decode]" \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.total_epochs=100 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "$@" 2>&1 | tee "${SMOKE_LOG}"

for role in prefill decode; do
    if ! grep -a -q -- "--disaggregation-mode ${role}" "${SMOKE_LOG}"; then
        echo "FAIL: no ${role} shard was launched. Log: ${SMOKE_LOG}" >&2
        exit 1
    fi
done
if ! grep -a -q -i "activating prefill router" "${VERL_DYNAMO_LOG_DIR}"/*frontend*.log; then
    echo "FAIL: the frontend never activated its prefill router. Logs: ${VERL_DYNAMO_LOG_DIR}" >&2
    exit 1
fi
pearson=$(grep -a -o "training/rollout_actor_probs_pearson_corr:[0-9.eE+-]*" "${SMOKE_LOG}" | tail -1 | cut -d: -f2 || true)
if [[ -z "${pearson}" ]] || ! awk -v p="${pearson}" -v m="${MIN_PEARSON}" 'BEGIN { exit !(p + 0 >= m + 0) }'; then
    echo "FAIL: rollout/actor logprob pearson='${pearson}' < MIN_PEARSON=${MIN_PEARSON}. Log: ${SMOKE_LOG}" >&2
    exit 1
fi
echo "PASS: Dynamo P/D 2-step training smoke completed (pearson=${pearson} >= ${MIN_PEARSON})"
