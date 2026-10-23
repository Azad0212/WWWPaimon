#!/bin/bash

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PY="${PROJECT_ROOT}/train.py"

SEQ_LEN=96
DATA_TYPE="etth2"
MODEL_NAME="pai"

PRED_LENS=(96 192 336 720)

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

get_config () {
    local pred_len=$1

    MODEL_ARGS=()

    case "${pred_len}" in
        96)
            MODEL_ARGS=(
                --dropout 0.5
            )
            ;;

        192)
            MODEL_ARGS=(
                --dropout 0.45
                --d_model 96
                --d_ff 384
            )
            ;;

        336)
            MODEL_ARGS=(
                --dropout 0.44
                --d_model 96
                --d_ff 384
            )
            ;;

        720)
            MODEL_ARGS=(
                --dropout 0.37
                --d_ff 384
                --affine 0
            )
            ;;

        *)
            echo "ERROR: unsupported pred_len=${pred_len}" >&2
            return 1
            ;;
    esac
}

run_task () {
    local pred_len=$1

    get_config "${pred_len}" || return 1

    local log_file="${LOG_DIR}/${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}_pred${pred_len}.log"

    echo "============================================================"
    echo "Running: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"
    echo "Arguments: ${MODEL_ARGS[*]}"
    echo "Log file: ${log_file}"
    echo "============================================================"

    : > "${log_file}"

    (
        cd "${PROJECT_ROOT}" || exit 1

        python3 "${TRAIN_PY}" \
            --model_name "${MODEL_NAME}" \
            --data_type "${DATA_TYPE}" \
            --seq_len "${SEQ_LEN}" \
            --pred_len "${pred_len}" \
            "${MODEL_ARGS[@]}"
    ) 2>&1 | tee "${log_file}"

    local status=${PIPESTATUS[0]}
    local test_line

    test_line=$(grep -F "[TEST]" "${log_file}" | tail -n 1)

    if [ -z "${test_line}" ]; then
        test_line=$(tail -n 1 "${log_file}")
    fi

    if [ "${status}" -eq 0 ]; then
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | ${test_line}" \
            >> "${RESULT_FILE}"

        echo "Finished: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"
    else
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | FAILED | ${test_line}" \
            >> "${RESULT_FILE}"

        echo "FAILED: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"
    fi

    echo "----------------------------------------"
}

for pred_len in "${PRED_LENS[@]}"; do
    run_task "${pred_len}"
done

echo "============================================================"
echo "All tasks finished."
echo "Results saved to: ${RESULT_FILE}"
echo "Logs saved to: ${LOG_DIR}"
echo "============================================================"