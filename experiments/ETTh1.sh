#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_TYPE="etth1"
MODEL_NAME="pai"
SEQ_LEN=96


PRED_LENS=(96 192 336 720)

DROPOUT=0.05
D_MODEL=16
DEPTH=1
PERIOD=24

LOG_DIR="${SCRIPT_DIR}/logs/logs_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}"
mkdir -p "${LOG_DIR}"

RESULT_DIR="${SCRIPT_DIR}/TXT"
mkdir -p "${RESULT_DIR}"

RESULT_FILE="${RESULT_DIR}/results_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}.txt"
> "${RESULT_FILE}"

get_config () {
    local pred_len=$1

    case "${pred_len}" in
        96)
            PATCH_LEN=12
            PATCH_STRIDE=6
            D_FF=48
            ;;
        192|336)
            PATCH_LEN=12
            PATCH_STRIDE=6
            D_FF=64
            ;;
        720)
            PATCH_LEN=24
            PATCH_STRIDE=12
            D_FF=96
            ;;
        *)
            echo "ERROR: Unsupported pred_len=${pred_len}"
            return 1
            ;;
    esac
}

run_task () {
    local pred_len=$1

    get_config "${pred_len}" || return 1

    local log_file="${LOG_DIR}/${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}_pred${pred_len}.log"

    echo "============================================================"
    echo "Running configuration:"
    echo "  model_name   = ${MODEL_NAME}"
    echo "  data_type    = ${DATA_TYPE}"
    echo "  seq_len      = ${SEQ_LEN}"
    echo "  pred_len     = ${pred_len}"
    echo "  patch_len    = ${PATCH_LEN}"
    echo "  patch_stride = ${PATCH_STRIDE}"
    echo "  d_model      = ${D_MODEL}"
    echo "  d_ff         = ${D_FF}"
    echo "  depth        = ${DEPTH}"
    echo "  period       = ${PERIOD}"
    echo "  dropout      = ${DROPOUT}"
    echo "============================================================"

    : > "${log_file}"

    cd "${PROJECT_ROOT}" || exit 1

    python3 train.py \
        --model_name "${MODEL_NAME}" \
        --data_type "${DATA_TYPE}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${pred_len}" \
        --patch_len "${PATCH_LEN}" \
        --patch_stride "${PATCH_STRIDE}" \
        --dropout "${DROPOUT}" \
        --d_model "${D_MODEL}" \
        --d_ff "${D_FF}" \
        --depth "${DEPTH}" \
        --period "${PERIOD}" \
        > "${log_file}" 2>&1

    local status=$?


    local test_line
    test_line=$(grep "\[TEST\]" "${log_file}" | tail -n 1)

    if [ -z "${test_line}" ]; then
        test_line=$(tail -n 1 "${log_file}")
    fi

    if [ ${status} -eq 0 ]; then
        echo "pred_len=${pred_len} | patch_len=${PATCH_LEN} | patch_stride=${PATCH_STRIDE} | d_ff=${D_FF} | ${test_line}" \
            | tee -a "${RESULT_FILE}"
    else
        echo "pred_len=${pred_len} | patch_len=${PATCH_LEN} | patch_stride=${PATCH_STRIDE} | d_ff=${D_FF} | FAILED | ${test_line}" \
            | tee -a "${RESULT_FILE}"
    fi

    echo "Log saved to: ${log_file}"
    echo "------------------------------------------------------------"
}

for pred_len in "${PRED_LENS[@]}"; do
    run_task "${pred_len}"
done

echo "All tasks finished."
echo "Test results saved to: ${RESULT_FILE}"
