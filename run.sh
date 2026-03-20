#!/bin/bash

export D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peng/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/home/peng/.mujoco/mujoco210
export MUJOCO_GL=egl

ENV_NAME="hopper-medium-v2"
GPU_ID=0

for SEED in 0 1 2
do
  LOG_DIR="./runs/physiql_clean_${ENV_NAME}_gpu${GPU_ID}_seed${SEED}"
  mkdir -p "${LOG_DIR}"

  nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" -u main.py \
    --log-dir "${LOG_DIR}" \
    --env-name "${ENV_NAME}" \
    --algo-name physiql \
    --model-mode separate \
    --use-forward 1 \
    --use-inverse 1 \
    --use-phys 1 \
    --corruption-type none \
    --corruption-ratio 0.0 \
    --corruption-std 0.0 \
    --tau 0.7 \
    --beta 3.0 \
    --seed "${SEED}" \
    --gpu-id 0 \
    --n-steps 1000000 \
    --eval-period 5000 \
    > "${LOG_DIR}/train.log" 2>&1 &
done