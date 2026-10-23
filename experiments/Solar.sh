#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PY="${PROJECT_ROOT}/train.py"

DATA_TYPE="solar"
MODEL_NAME="pai"
SEQ_LEN=336

PRED_LENS=(96 192 336 720)

PERIOD=144
D_FF=512

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
            D_MODEL=96
            DROPOUT=0.05

            MODEL_ARGS=(
                --period "${PERIOD}"
                --d_ff "${D_FF}"
                --d_model "${D_MODEL}"
                --dropout "${DROPOUT}"
            )
            ;;

        192)
            D_MODEL=96
            DROPOUT=0.005

            MODEL_ARGS=(
                --period "${PERIOD}"
                --d_ff "${D_FF}"
                --d_model "${D_MODEL}"
                --dropout "${DROPOUT}"
            )
            ;;

        336|720)
            PATCH_LEN=24
            PATCH_STRIDE=12
            D_MODEL=128
            ENC_LAYERS=2
            DROPOUT=0.005

            MODEL_ARGS=(
                --patch_len "${PATCH_LEN}"
                --patch_stride "${PATCH_STRIDE}"
                --period "${PERIOD}"
                --d_model "${D_MODEL}"
                --d_ff "${D_FF}"
                --enc_layers "${ENC_LAYERS}"
                --dropout "${DROPOUT}"
            )
            ;;

        *)
            echo "ERROR: Unsupported pred_len=${pred_len}" >&2
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
    echo "  model_name = ${MODEL_NAME}"
    echo "  data_type  = ${DATA_TYPE}"
    echo "  seq_len    = ${SEQ_LEN}"
    echo "  pred_len   = ${pred_len}"
    echo "  arguments  = ${MODEL_ARGS[*]}"
    echo "  log_file   = ${log_file}"
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
    ) > "${log_file}" 2>&1

    local status=$?
    local test_line

    test_line=$(grep -F "[TEST]" "${log_file}" | tail -n 1)

    if [ -z "${test_line}" ]; then
        test_line=$(tail -n 1 "${log_file}")
    fi

    if [ "${status}" -eq 0 ]; then
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | ${MODEL_ARGS[*]} | ${test_line}" \
            | tee -a "${RESULT_FILE}"

        echo "Finished: pred_len=${pred_len}"
    else
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | ${MODEL_ARGS[*]} | FAILED | ${test_line}" \
            | tee -a "${RESULT_FILE}"

        echo "FAILED: pred_len=${pred_len}"
    fi

    echo "Log saved to: ${log_file}"
    echo "------------------------------------------------------------"
}

for pred_len in "${PRED_LENS[@]}"; do
    run_task "${pred_len}"
done

echo "============================================================"
echo "All tasks finished."
echo "Results saved to: ${RESULT_FILE}"
echo "============================================================"