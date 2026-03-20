#!/bin/bash

export D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peng/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/home/peng/.mujoco/mujoco210
export MUJOCO_GL=egl

ENV_NAME="hopper-medium-v2"
PYTHON_BIN=/opt/conda/miniconda3/envs/iql/bin/python

WANDB_ENTITY="dlut-pqj"
WANDB_PROJECT="A10-iql"

N_STEPS=1000000
EVAL_PERIOD=5000

TAU=0.7
BETA=3.0

FIXED_CORRUPTION=1
CORRUPTION_SEED=123

SEEDS=(2 5 7)

# 三组实验分配到不同 GPU
GPU_CLEAN=0
GPU_BASELINE=1
GPU_OURS=2

# ============================================
# Group 1: Clean + PhysIQL
# ============================================
for SEED in "${SEEDS[@]}"
do
  LOG_DIR="./runs/stage1_clean_physiql_${ENV_NAME}_gpu${GPU_CLEAN}_seed${SEED}"
  mkdir -p "${LOG_DIR}"

  nohup env CUDA_VISIBLE_DEVICES="${GPU_CLEAN}" "${PYTHON_BIN}" -u main.py \
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
    --fixed-corruption "${FIXED_CORRUPTION}" \
    --corruption-seed "${CORRUPTION_SEED}" \
    --tau "${TAU}" \
    --beta "${BETA}" \
    --seed "${SEED}" \
    --gpu-id 0 \
    --n-steps "${N_STEPS}" \
    --eval-period "${EVAL_PERIOD}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --core-log-interval 100 \
    --analysis-log-interval 5000 \
    > "${LOG_DIR}/train.log" 2>&1 &
done

# ============================================
# Group 2: Fixed Corruption + IQL baseline
# ============================================
for SEED in "${SEEDS[@]}"
do
  LOG_DIR="./runs/stage1_fixedcorr_iql_${ENV_NAME}_gpu${GPU_BASELINE}_seed${SEED}"
  mkdir -p "${LOG_DIR}"

  nohup env CUDA_VISIBLE_DEVICES="${GPU_BASELINE}" "${PYTHON_BIN}" -u main.py \
    --log-dir "${LOG_DIR}" \
    --env-name "${ENV_NAME}" \
    --algo-name iql \
    --model-mode separate \
    --use-forward 0 \
    --use-inverse 0 \
    --use-phys 0 \
    --corruption-type transition_shuffle \
    --corruption-ratio 0.1 \
    --corruption-std 0.0 \
    --fixed-corruption "${FIXED_CORRUPTION}" \
    --corruption-seed "${CORRUPTION_SEED}" \
    --tau "${TAU}" \
    --beta "${BETA}" \
    --seed "${SEED}" \
    --gpu-id 0 \
    --n-steps "${N_STEPS}" \
    --eval-period "${EVAL_PERIOD}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --core-log-interval 100 \
    --analysis-log-interval 5000 \
    > "${LOG_DIR}/train.log" 2>&1 &
done

# ============================================
# Group 3: Fixed Corruption + PhysIQL (Ours)
# ============================================
for SEED in "${SEEDS[@]}"
do
  LOG_DIR="./runs/stage1_fixedcorr_physiql_${ENV_NAME}_gpu${GPU_OURS}_seed${SEED}"
  mkdir -p "${LOG_DIR}"

  nohup env CUDA_VISIBLE_DEVICES="${GPU_OURS}" "${PYTHON_BIN}" -u main.py \
    --log-dir "${LOG_DIR}" \
    --env-name "${ENV_NAME}" \
    --algo-name physiql \
    --model-mode separate \
    --use-forward 1 \
    --use-inverse 1 \
    --use-phys 1 \
    --corruption-type transition_shuffle \
    --corruption-ratio 0.1 \
    --corruption-std 0.0 \
    --fixed-corruption "${FIXED_CORRUPTION}" \
    --corruption-seed "${CORRUPTION_SEED}" \
    --tau "${TAU}" \
    --beta "${BETA}" \
    --seed "${SEED}" \
    --gpu-id 0 \
    --n-steps "${N_STEPS}" \
    --eval-period "${EVAL_PERIOD}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --core-log-interval 100 \
    --analysis-log-interval 5000 \
    > "${LOG_DIR}/train.log" 2>&1 &
done