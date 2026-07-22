#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# 为一个 GPU slot 启动长期运行的 Policy Server
#
# 主脚本调用形式：
#
# CUDA_VISIBLE_DEVICES=<physical_gpu> \
# bash run_policy_server.sh \
#     <checkpoint_path> \
#     <host> \
#     <port>
#
# CUDA_VISIBLE_DEVICES 设置后，Python 进程内部只能看到一张 GPU，
# 因此服务端代码继续使用 cuda:0 即可。
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_starvla_root() {
    local current="${SCRIPT_DIR}"

    while [[ "${current}" != "/" ]]; do
        if [[ -f "${current}/deployment/model_server/server_policy.py" ]] &&
           [[ -d "${current}/examples/simBenchmarks/LIBERO-plus" ]]; then

            printf '%s\n' "${current}"
            return 0
        fi

        current="$(dirname "${current}")"
    done

    return 1
}

# 优先使用显式设置的 STARVLA_DIR。
if [[ -n "${STARVLA_DIR:-}" ]]; then
    STARVLA_DIR="$(cd "${STARVLA_DIR}" && pwd)"

else
    if ! STARVLA_DIR="$(find_starvla_root)"; then
        echo "[ERROR] Cannot locate the StarVLA repository root." >&2
        echo "        Set STARVLA_DIR=/absolute/path/to/StarVLA." >&2
        exit 1
    fi
fi

# 兼容你原来的 ABot_python 环境变量。
ABOT_PYTHON="${ABot_python:-${ABOT_PYTHON:-python}}"

# 主接口使用位置参数，同时兼容原来的环境变量。
CKPT_PATH="${1:-${your_ckpt:-}}"
HOST="${2:-${server_host:-127.0.0.1}}"
PORT="${3:-${base_port:-9883}}"

USE_BF16="${USE_BF16:-1}"

SERVER_ENTRY="${
    POLICY_SERVER_ENTRY:-
    deployment/model_server/server_policy.py
}"

# 你的原始 server_policy.py 没有传入 host。
#
# 服务端支持 host 参数时，可设置：
#
# export POLICY_SERVER_HOST_FLAG=--host
#
# 默认保持为空，不向 Python 传 host。
POLICY_SERVER_HOST_FLAG="${POLICY_SERVER_HOST_FLAG:-}"

# 其他额外参数，例如：
#
# export POLICY_SERVER_EXTRA_ARGS='--action_horizon 8'
POLICY_SERVER_EXTRA_ARGS="${POLICY_SERVER_EXTRA_ARGS:-}"

if [[ -z "${CKPT_PATH}" ]]; then
    echo "[ERROR] Missing checkpoint path." >&2
    echo \
        "Usage: bash run_policy_server.sh" \
        "<checkpoint_path> <host> <port>" >&2
    exit 1
fi

if [[ ! -e "${CKPT_PATH}" ]]; then
    echo "[ERROR] Checkpoint does not exist: ${CKPT_PATH}" >&2
    exit 1
fi

if [[ ! -f "${STARVLA_DIR}/${SERVER_ENTRY}" ]]; then
    echo "[ERROR] Policy server entry does not exist:" >&2
    echo "        ${STARVLA_DIR}/${SERVER_ENTRY}" >&2
    exit 1
fi

if ! command -v "${ABOT_PYTHON}" >/dev/null 2>&1 &&
   [[ ! -x "${ABOT_PYTHON}" ]]; then

    echo "[ERROR] Python executable not found: ${ABOT_PYTHON}" >&2
    exit 1
fi

# 单独执行本脚本时，仍兼容原来的 gpu_id。
#
# 由主调度器调用时，CUDA_VISIBLE_DEVICES 已经设置。
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu_id:-0}"
fi

cd "${STARVLA_DIR}"

export PYTHONPATH="${
    STARVLA_DIR
}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
    "${ABOT_PYTHON}"
    "${SERVER_ENTRY}"

    --ckpt_path
    "${CKPT_PATH}"

    --port
    "${PORT}"
)

# 只有服务端代码确实支持 host 时才添加。
if [[ -n "${POLICY_SERVER_HOST_FLAG}" ]]; then
    CMD+=(
        "${POLICY_SERVER_HOST_FLAG}"
        "${HOST}"
    )
fi

if [[ "${USE_BF16}" == "1" ]]; then
    CMD+=(--use_bf16)
fi

if [[ -n "${POLICY_SERVER_EXTRA_ARGS}" ]]; then
    read -r -a EXTRA_ARGS <<< "${POLICY_SERVER_EXTRA_ARGS}"
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "[POLICY SERVER]"
echo "repo                 : ${STARVLA_DIR}"
echo "python               : ${ABOT_PYTHON}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "checkpoint           : ${CKPT_PATH}"
echo "host                 : ${HOST}"
echo "port                 : ${PORT}"
echo "bf16                 : ${USE_BF16}"

printf 'command              :'
printf ' %q' "${CMD[@]}"
printf '\n'

echo "============================================================"

# 使用 exec 后，当前 Bash 进程会被 Python Server 替换。
#
# 主调度器记录到的 PID 就是实际 Server PID，
# 便于监控和清理。
exec "${CMD[@]}"