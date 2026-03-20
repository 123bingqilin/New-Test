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

for SEED in 1 2 5
do
  LOG_DIR="./runs/iql_clean_${ENV_NAME}_gpu${GPU_ID}_seed${SEED}"
  mkdir -p "${LOG_DIR}"

  nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u main.py \
    --log-dir "${LOG_DIR}" \
    --env-name "${ENV_NAME}" \
    --algo-name iql \
    --model-mode separate \
    --use-forward 0 \
    --use-inverse 0 \
    --use-phys 0 \
    --corruption-type none \
    --corruption-ratio 0.0 \
    --corruption-std 0.0 \
    --tau 0.7 \
    --beta 3.0 \
    --seed "${SEED}" \
    --gpu-id 0 \
    --n-steps 1000000 \
    --eval-period 5000 \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --core-log-interval 100 \
    --analysis-log-interval 5000 \
    > "${LOG_DIR}/train.log" 2>&1 &
done