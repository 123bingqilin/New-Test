#!/bin/bash
set -euo pipefail

export D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/peng/.mujoco/mujoco210/bin"
export MUJOCO_PY_MUJOCO_PATH="/home/peng/.mujoco/mujoco210"
export MUJOCO_GL=egl

ENV_NAME="hopper-medium-v2"
PYTHON_BIN="/opt/conda/miniconda3/envs/iql/bin/python"
WANDB_ENTITY="dlut-pqj"
WANDB_PROJECT="A10-iql"
BASE_LOG_DIR="./runs"

SEEDS=(2 5 7)
GPUS=(0 1 2)

# 切到纯 IQL 时改成: ALGO_NAME="iql"
ALGO_NAME="physiql"
MODEL_MODE="separate"
USE_FORWARD=1
USE_INVERSE=1
USE_PHYS=1
USE_GLOBAL_CONF=1

CORRUPTION_TYPE="transition_shuffle"
CORRUPTION_RATIO=0.10
CORRUPTION_STD=0.0
FIXED_CORRUPTION=1
CORRUPTION_SEED=0

LAMBDA_F=1.0
LAMBDA_INV=1.0
AUX_WEIGHT=1.0
PHYS_WEIGHT_MIN=0.05
PHYS_DELTA=0.0
PHYS_TAU=1.0
PHYS_MAD_EPS=1e-6
PHYS_GLOBAL_LAMBDA=1.0
PHYS_GLOBAL_RHO=0.99
PHYS_GLOBAL_ETA=0.5
PHYS_GLOBAL_INIT=1.0
PHYS_WARMUP_START=50000
PHYS_WARMUP_END=200000

CORE_LOG_INTERVAL=100
ANALYSIS_LOG_INTERVAL=5000
EVAL_PERIOD=5000
N_STEPS=1000000
BATCH_SIZE=256

if [ ${#SEEDS[@]} -ne ${#GPUS[@]} ]; then
  echo "SEEDS and GPUS must have the same length."
  exit 1
fi

mkdir -p "${BASE_LOG_DIR}"

for idx in "${!SEEDS[@]}"; do
  SEED="${SEEDS[$idx]}"
  GPU_ID="${GPUS[$idx]}"

  RUN_TAG="${ALGO_NAME}_${ENV_NAME}_f${USE_FORWARD}_i${USE_INVERSE}_p${USE_PHYS}_g${USE_GLOBAL_CONF}_seed${SEED}_gpu${GPU_ID}"
  LOG_DIR="${BASE_LOG_DIR}/${RUN_TAG}"
  OUT_LOG="${LOG_DIR}/train.log"
  mkdir -p "${LOG_DIR}"

  echo "Launching ${RUN_TAG} on GPU ${GPU_ID}"

  nohup "${PYTHON_BIN}" -u main.py     --log-dir "${LOG_DIR}"     --env-name "${ENV_NAME}"     --seed "${SEED}"     --gpu-id "${GPU_ID}"     --algo-name "${ALGO_NAME}"     --model-mode "${MODEL_MODE}"     --use-forward "${USE_FORWARD}"     --use-inverse "${USE_INVERSE}"     --use-phys "${USE_PHYS}"     --use-global-conf "${USE_GLOBAL_CONF}"     --n-steps "${N_STEPS}"     --batch-size "${BATCH_SIZE}"     --eval-period "${EVAL_PERIOD}"     --core-log-interval "${CORE_LOG_INTERVAL}"     --analysis-log-interval "${ANALYSIS_LOG_INTERVAL}"     --wandb-entity "${WANDB_ENTITY}"     --wandb-project "${WANDB_PROJECT}"     --corruption-type "${CORRUPTION_TYPE}"     --corruption-ratio "${CORRUPTION_RATIO}"     --corruption-std "${CORRUPTION_STD}"     --fixed-corruption "${FIXED_CORRUPTION}"     --corruption-seed "${CORRUPTION_SEED}"     --lambda-f "${LAMBDA_F}"     --lambda-inv "${LAMBDA_INV}"     --aux-weight "${AUX_WEIGHT}"     --phys-weight-min "${PHYS_WEIGHT_MIN}"     --phys-delta "${PHYS_DELTA}"     --phys-tau "${PHYS_TAU}"     --phys-mad-eps "${PHYS_MAD_EPS}"     --phys-global-lambda "${PHYS_GLOBAL_LAMBDA}"     --phys-global-rho "${PHYS_GLOBAL_RHO}"     --phys-global-eta "${PHYS_GLOBAL_ETA}"     --phys-global-init "${PHYS_GLOBAL_INIT}"     --phys-warmup-start "${PHYS_WARMUP_START}"     --phys-warmup-end "${PHYS_WARMUP_END}"     > "${OUT_LOG}" 2>&1 &
done

echo "All jobs launched. Use 'nvidia-smi' and 'tail -f runs/.../train.log' to monitor."
