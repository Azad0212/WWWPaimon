#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PY="${PROJECT_ROOT}/train.py"

SEQ_LEN=336
DATA_TYPE="ecl"
MODEL_NAME="pai"

PRED_LENS=(96 192 336 720)

PATCH_LEN=24
PATCH_STRIDE=12
PERIOD=168
D_MODEL=128
D_FF=512
N_HEADS=4
FILTER_KERNELS="auto"

LOG_DIR="${SCRIPT_DIR}/logs/logs_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}"
mkdir -p "${LOG_DIR}"

RESULT_DIR="${SCRIPT_DIR}/TXT"
mkdir -p "${RESULT_DIR}"

RESULT_FILE="${RESULT_DIR}/results_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}.txt"
> "${RESULT_FILE}"

if [ ! -f "${TRAIN_PY}" ]; then
    echo "ERROR: train.py not found at ${TRAIN_PY}"
    exit 1
fi

get_config() {
    local pred_len=$1

    case "${pred_len}" in
        96)
            ENC_LAYERS=3
            DROPOUT=0.10
            ;;
        192)
            ENC_LAYERS=4
            DROPOUT=0.005
            ;;
        336)
            ENC_LAYERS=4
            DROPOUT=0.05
            ;;
        720)
            ENC_LAYERS=4
            DROPOUT=0.005
            ;;
        *)
            echo "ERROR: Unsupported pred_len=${pred_len}"
            return 1
            ;;
    esac
}

run_task() {
    local pred_len=$1

    get_config "${pred_len}" || return 1

    local log_file="${LOG_DIR}/${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}_pred${pred_len}.log"

    echo "============================================================"
    echo "Running: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"
    echo "Config:"
    echo "  patch_len=${PATCH_LEN}"
    echo "  patch_stride=${PATCH_STRIDE}"
    echo "  period=${PERIOD}"
    echo "  filter_kernels=${FILTER_KERNELS}"
    echo "  d_model=${D_MODEL}"
    echo "  d_ff=${D_FF}"
    echo "  enc_layers=${ENC_LAYERS}"
    echo "  n_heads=${N_HEADS}"
    echo "  dropout=${DROPOUT}"
    echo "Log: ${log_file}"
    echo "============================================================"

    : > "${log_file}"

    cd "${PROJECT_ROOT}" || exit 1

    python3 "${TRAIN_PY}" \
        --model_name "${MODEL_NAME}" \
        --data_type "${DATA_TYPE}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${pred_len}" \
        --patch_len "${PATCH_LEN}" \
        --patch_stride "${PATCH_STRIDE}" \
        --period "${PERIOD}" \
        --filter_kernels "${FILTER_KERNELS}" \
        --d_model "${D_MODEL}" \
        --d_ff "${D_FF}" \
        --enc_layers "${ENC_LAYERS}" \
        --n_heads "${N_HEADS}" \
        --dropout "${DROPOUT}" \
        > "${log_file}" 2>&1

    local status=$?
    local last_line

    last_line=$(tail -n 1 "${log_file}")

    if [ ${status} -eq 0 ]; then
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | patch_len=${PATCH_LEN} | patch_stride=${PATCH_STRIDE} | period=${PERIOD} | filter_kernels=${FILTER_KERNELS} | d_model=${D_MODEL} | d_ff=${D_FF} | enc_layers=${ENC_LAYERS} | n_heads=${N_HEADS} | dropout=${DROPOUT} | ${last_line}" \
            | tee -a "${RESULT_FILE}"
    else
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | patch_len=${PATCH_LEN} | patch_stride=${PATCH_STRIDE} | period=${PERIOD} | filter_kernels=${FILTER_KERNELS} | d_model=${D_MODEL} | d_ff=${D_FF} | enc_layers=${ENC_LAYERS} | n_heads=${N_HEADS} | dropout=${DROPOUT} | FAILED | ${last_line}" \
            | tee -a "${RESULT_FILE}"
    fi

    echo "----------------------------------------"
}

for pred_len in "${PRED_LENS[@]}"; do
    run_task "${pred_len}"
done

echo "All tasks finished."
echo "Results saved to: ${RESULT_FILE}"