#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# 执行一个 LIBERO-plus 切片
#
# 主脚本调用形式：
#
# CUDA_VISIBLE_DEVICES=<physical_gpu> \
# LIBERO_PRETRAINED_PATH=<checkpoint> \
# bash eval_libero_in_one.sh \
#     <task_suite_name> \
#     <start_idx> \
#     <end_idx> \
#     <server_host> \
#     <server_port> \
#     <result_dir>
#
# 本脚本：
#
# 1. 不启动 Policy Server；
# 2. 不再次切分 start/end；
# 3. 不在内部启动多个后台 evaluator；
# 4. 只连接主调度器为当前 GPU 启动的长期 Server。
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_starvla_root() {
    local current="${SCRIPT_DIR}"

    while [[ "${current}" != "/" ]]; do
        if [[ -f "${
                current
            }/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py" ]] &&
           [[ -d "${current}/deployment/model_server" ]]; then

            printf '%s\n' "${current}"
            return 0
        fi

        current="$(dirname "${current}")"
    done

    return 1
}

if [[ -n "${STARVLA_DIR:-}" ]]; then
    STARVLA_DIR="$(cd "${STARVLA_DIR}" && pwd)"

else
    if ! STARVLA_DIR="$(find_starvla_root)"; then
        echo "[ERROR] Cannot locate the StarVLA repository root." >&2
        echo "        Set STARVLA_DIR=/absolute/path/to/StarVLA." >&2
        exit 1
    fi
fi

LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"

# 保留你原来的默认渲染方式。
MUJOCO_GL="${MUJOCO_GL:-osmesa}"

TASK_SUITE_NAME="${1:?Missing task_suite_name}"
START_IDX="${2:?Missing start_idx}"
END_IDX="${3:?Missing end_idx}"

SERVER_HOST="${4:-127.0.0.1}"
SERVER_PORT="${5:?Missing server_port}"

OUTPUT_DIR="${
    6:-
    ${output_dir:-${STARVLA_DIR}/results/libero_plus_parallel_eval}
}"

# 主调度器通过 LIBERO_PRETRAINED_PATH 传入。
#
# 同时兼容你原来的 your_ckpt 环境变量。
PRETRAINED_PATH="${
    LIBERO_PRETRAINED_PATH:-
    ${your_ckpt:-}
}"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"

EVAL_ENTRY="${
    LIBERO_EVAL_ENTRY:-
    examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py
}"

# eval_libero_model.py 中连接 Model Server 的参数名。
#
# 默认假设为：
#
# --host
# --port
#
# 实际参数不同时，可以通过环境变量修改。
EVAL_HOST_FLAG="${LIBERO_EVAL_HOST_FLAG:---host}"
EVAL_PORT_FLAG="${LIBERO_EVAL_PORT_FLAG:---port}"

LIBERO_EVAL_EXTRA_ARGS="${LIBERO_EVAL_EXTRA_ARGS:-}"

if [[ -z "${LIBERO_HOME}" ]]; then
    echo "[ERROR] LIBERO_HOME is required." >&2
    exit 1
fi

if [[ -z "${PRETRAINED_PATH}" ]]; then
    echo "[ERROR] LIBERO_PRETRAINED_PATH is required." >&2
    exit 1
fi

if [[ ! -e "${PRETRAINED_PATH}" ]]; then
    echo \
        "[ERROR] Pretrained path does not exist:" \
        "${PRETRAINED_PATH}" >&2

    exit 1
fi

if [[ ! -f "${STARVLA_DIR}/${EVAL_ENTRY}" ]]; then
    echo "[ERROR] Evaluation entry does not exist:" >&2
    echo "        ${STARVLA_DIR}/${EVAL_ENTRY}" >&2
    exit 1
fi

if ! [[ "${START_IDX}" =~ ^[0-9]+$ ]] ||
   ! [[ "${END_IDX}" =~ ^[0-9]+$ ]] ||
   (( START_IDX >= END_IDX )); then

    echo \
        "[ERROR] Invalid interval:" \
        "[${START_IDX}, ${END_IDX})" >&2

    exit 1
fi

if ! command -v "${LIBERO_PYTHON}" >/dev/null 2>&1 &&
   [[ ! -x "${LIBERO_PYTHON}" ]]; then

    echo \
        "[ERROR] Python executable not found:" \
        "${LIBERO_PYTHON}" >&2

    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${STARVLA_DIR}"

export MUJOCO_GL
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

export PYTHONPATH="${
    LIBERO_HOME
}:${STARVLA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# 同时通过环境变量暴露 Server 地址。
#
# 当 eval_libero_model.py 不使用 CLI 参数，而是读取环境变量时，
# 可以直接读取这两个变量。
export LIBERO_SERVER_HOST="${SERVER_HOST}"
export LIBERO_SERVER_PORT="${SERVER_PORT}"

CMD=(
    "${LIBERO_PYTHON}"
    "${EVAL_ENTRY}"

    --pretrained_path
    "${PRETRAINED_PATH}"

    --task_suite_name
    "${TASK_SUITE_NAME}"

    --num_trials_per_task
    "${NUM_TRIALS_PER_TASK}"

    --output_dir
    "${OUTPUT_DIR}"

    --start_idx
    "${START_IDX}"

    --end_idx
    "${END_IDX}"
)

if [[ -n "${EVAL_HOST_FLAG}" ]]; then
    CMD+=(
        "${EVAL_HOST_FLAG}"
        "${SERVER_HOST}"
    )
fi

if [[ -n "${EVAL_PORT_FLAG}" ]]; then
    CMD+=(
        "${EVAL_PORT_FLAG}"
        "${SERVER_PORT}"
    )
fi

if [[ -n "${LIBERO_EVAL_EXTRA_ARGS}" ]]; then
    read -r -a EXTRA_ARGS <<< "${LIBERO_EVAL_EXTRA_ARGS}"
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "[LIBERO EVALUATOR]"
echo "repo                 : ${STARVLA_DIR}"
echo "python               : ${LIBERO_PYTHON}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "MUJOCO_GL            : ${MUJOCO_GL}"
echo "task suite           : ${TASK_SUITE_NAME}"
echo "range                : [${START_IDX}, ${END_IDX})"
echo "server               : ${SERVER_HOST}:${SERVER_PORT}"
echo "checkpoint           : ${PRETRAINED_PATH}"
echo "output               : ${OUTPUT_DIR}"

printf 'command              :'
printf ' %q' "${CMD[@]}"
printf '\n'

echo "============================================================"

# 一个 slot 只执行一个 evaluator。
#
# exec 保证主调度器记录的 PID 就是实际 Python evaluator PID。
exec "${CMD[@]}"