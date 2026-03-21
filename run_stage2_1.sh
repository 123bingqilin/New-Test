#!/bin/bash

export D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peng/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/home/peng/.mujoco/mujoco210
export MUJOCO_GL=egl

ENV_NAME="hopper-medium-v2"
GPU_ID=3
PYTHON_BIN=/opt/conda/miniconda3/envs/iql/bin/python
WANDB_ENTITY="dlut-pqj"
WANDB_PROJECT="A10-iql"

PHYS_NORM_MODE="none"      # none / std / quantile
PHYS_Q_CENTER=0.50
PHYS_Q_UPPER=0.95

for SEED in 2 5 7
do
  LOG_DIR="./runs/physiql_${PHYS_NORM_MODE}_${ENV_NAME}_gpu${GPU_ID}_seed${SEED}"
  mkdir -p "${LOG_DIR}"

  nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u main.py \
    --log-dir "${LOG_DIR}" \
    --env-name "${ENV_NAME}" \
    --seed "${SEED}" \
    --algo-name physiql \
    --model-mode separate \
    --use-forward 1 \
    --use-inverse 1 \
    --use-phys 1 \
    --lambda-inv 1.0 \
    --alpha-phys 0.1 \
    --aux-weight 1.0 \
    --phys-norm-mode "${PHYS_NORM_MODE}" \
    --phys-quantile-center "${PHYS_Q_CENTER}" \
    --phys-quantile-upper "${PHYS_Q_UPPER}" \
    --corruption-type transition_shuffle \
    --corruption-ratio 0.1 \
    --fixed-corruption 1 \
    --corruption-seed 123 \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    > "${LOG_DIR}/train.log" 2>&1 &
done