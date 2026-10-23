#!/bin/bash



SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEQ_LEN=96
DATA_TYPE="cernet"
MODEL_NAME="pai"

PRED_LENS=(96 192 336 720)

LOG_DIR="${SCRIPT_DIR}/logs/logs_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}"
mkdir -p "${LOG_DIR}"

RESULT_DIR="${SCRIPT_DIR}/TXT"
mkdir -p "${RESULT_DIR}"

RESULT_FILE="${RESULT_DIR}/results_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}.txt"
> "${RESULT_FILE}"

run_task () {
    local pred_len=$1
    local log_file="${LOG_DIR}/${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}_pred${pred_len}.log"

    echo "Running: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"

    : > "${log_file}"

    cd "${PROJECT_ROOT}" || exit 1

    python3 train.py \
        --model_name "${MODEL_NAME}" \
        --data_type "${DATA_TYPE}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${pred_len}" \
        --period 128 \
        --d_ff 256 \
        --d_model 128 \
        --dropout 0 \
        2>&1 | tee "${log_file}"

    STATUS=${PIPESTATUS[0]}
    LAST_LINE=$(tail -n 1 "${log_file}")

    if [ ${STATUS} -eq 0 ]; then
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | ${LAST_LINE}" >> "${RESULT_FILE}"
        echo "Finished: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"
    else
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | FAILED | ${LAST_LINE}" >> "${RESULT_FILE}"
        echo "FAILED: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"
    fi

    echo "----------------------------------------"
}

for pred_len in "${PRED_LENS[@]}"; do
    run_task "${pred_len}"
done

echo "All tasks finished."
echo "Results saved to: ${RESULT_FILE}"