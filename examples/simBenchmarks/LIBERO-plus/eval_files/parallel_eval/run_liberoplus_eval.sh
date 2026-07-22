#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# LIBERO-plus 本地多 GPU 动态调度器
#
# 一个 slot 包含：
#   一张物理 GPU
#   一个长期运行的 Policy Server
#   一个独立 WebSocket 端口
#
# 每个 slot 同时执行一个 evaluator。切片完成后，不重新加载模型，
# 而是复用当前 slot 的 Policy Server，继续执行下一个切片。
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVER_SCRIPT="${LIBERO_SERVER_SCRIPT:-${SCRIPT_DIR}/run_policy_server.sh}"
EVAL_SCRIPT="${LIBERO_EVAL_SCRIPT:-${SCRIPT_DIR}/eval_libero_in_one.sh}"

# LIBERO-plus 的四个 benchmark。
BENCHMARKS=(
    "libero_10"
    "libero_goal"
    "libero_object"
    "libero_spatial"
)

# 对应原 Nebula 脚本中的总索引数量。
BENCHMARK_SIZES=(
    2519
    2591
    2518
    2402
)

HOST="${LIBERO_HOST:-127.0.0.1}"

# 与你原 run_policy_server.sh 的默认端口一致。
BASE_PORT="${LIBERO_BASE_PORT:-9883}"

# 每个 benchmark 默认切成 4 份。
SLICES_PER_BENCHMARK="${LIBERO_SLICES_PER_BENCHMARK:-4}"

SERVER_TIMEOUT="${LIBERO_SERVER_TIMEOUT:-600}"
POLL_INTERVAL="${LIBERO_POLL_INTERVAL:-3}"

LOG_ROOT="${LIBERO_LOG_ROOT:-${SCRIPT_DIR}/logs}"
RESULT_ROOT="${LIBERO_RESULT_ROOT:-${SCRIPT_DIR}/results}"

CKPT_PATH=""
REQUESTED_BENCHMARKS=()

# ============================================================================
# 任务队列
#
# 相同下标共同表示一个任务：
#
# JOB_TASKS[i]
# JOB_STARTS[i]
# JOB_ENDS[i]
# JOB_SLICE_IDS[i]
#
# 例如：
#   JOB_TASKS[0]    = libero_10
#   JOB_STARTS[0]   = 0
#   JOB_ENDS[0]     = 630
#   JOB_SLICE_IDS[0]= 0
# ============================================================================

JOB_TASKS=()
JOB_STARTS=()
JOB_ENDS=()
JOB_SLICE_IDS=()

# ============================================================================
# GPU slot 状态
# ============================================================================

SLOT_GPUS=()
SLOT_PORTS=()
SLOT_SERVER_PIDS=()
SLOT_SERVER_LOGS=()

# 每个 slot 当前运行的 evaluator。
ACTIVE_EVAL_PIDS=()
ACTIVE_JOB_IDS=()
ACTIVE_EVAL_LOGS=()

USED_PORTS=()
FAILED_JOBS=()

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR=""
RUN_RESULT_DIR=""

usage() {
    cat <<'USAGE'
Usage:
  bash start_libero_eval.sh --ckpt PATH [options] [all|benchmark ...]

Required:
  -c, --ckpt PATH
      Policy checkpoint path

Options:
  -s, --slices N
      Number of slices per benchmark
      Default: 4

  -p, --base-port PORT
      First Policy Server port
      Default: 9883

      --host HOST
      Policy Server host
      Default: 127.0.0.1

      --server-timeout SEC
      Maximum waiting time for each server
      Default: 600

      --poll-interval SEC
      Scheduler process polling interval
      Default: 3

      --log-root DIR
      Log root directory

      --result-root DIR
      Evaluation result root directory

  -h, --help
      Show help

Benchmarks:
  all
  libero_10
  libero_goal
  libero_object
  libero_spatial

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash start_libero_eval.sh \
      --ckpt /path/to/checkpoint.pt \
      --slices 4 \
      all
USAGE
}

trim() {
    local value="$1"

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    printf '%s\n' "${value}"
}

join_by() {
    local separator="$1"
    shift

    local output=""
    local item=""

    for item in "$@"; do
        output="${output:+${output}${separator}}${item}"
    done

    printf '%s' "${output}"
}

benchmark_size() {
    local requested="$1"
    local i

    for i in "${!BENCHMARKS[@]}"; do
        if [[ "${BENCHMARKS[$i]}" == "${requested}" ]]; then
            printf '%s\n' "${BENCHMARK_SIZES[$i]}"
            return 0
        fi
    done

    return 1
}

# ============================================================================
# GPU 检测
#
# 优先读取 CUDA_VISIBLE_DEVICES。
# 未设置时，通过 nvidia-smi 自动检测全部 GPU。
# ============================================================================

detect_cuda_devices() {
    local -a raw_devices=()
    local -a devices=()

    local gpu_count=""
    local device=""
    local i

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra raw_devices <<< "${CUDA_VISIBLE_DEVICES}"

    elif command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count="$(
            nvidia-smi --list-gpus 2>/dev/null |
            wc -l |
            tr -d ' '
        )"

        for ((i = 0; i < gpu_count; ++i)); do
            raw_devices+=("${i}")
        done
    fi

    for device in "${raw_devices[@]-}"; do
        device="$(trim "${device}")"

        if [[ -n "${device}" ]]; then
            devices+=("${device}")
        fi
    done

    if (( ${#devices[@]} == 0 )); then
        echo "[ERROR] No CUDA devices detected." >&2
        return 1
    fi

    printf '%s\n' "${devices[@]}"
}

# ============================================================================
# 端口管理
# ============================================================================

port_in_use() {
    local host="$1"
    local port="$2"

    if command -v ss >/dev/null 2>&1; then
        ss -tuln 2>/dev/null |
            grep -qE "[:.]${port}[[:space:]]"

    elif command -v lsof >/dev/null 2>&1; then
        lsof \
            -iTCP:"${port}" \
            -sTCP:LISTEN \
            >/dev/null 2>&1

    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln 2>/dev/null |
            grep -qE "[:.]${port}[[:space:]]"

    else
        python - "${host}" "${port}" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)

try:
    result = sock.connect_ex((host, port))
finally:
    sock.close()

raise SystemExit(0 if result == 0 else 1)
PY
    fi
}

port_reserved() {
    local requested="$1"
    local item=""

    for item in "${USED_PORTS[@]-}"; do
        if [[ "${item}" == "${requested}" ]]; then
            return 0
        fi
    done

    return 1
}

find_available_port() {
    local port="$1"

    while port_reserved "${port}" ||
          port_in_use "${HOST}" "${port}"; do
        port=$((port + 1))
    done

    USED_PORTS+=("${port}")

    printf '%s\n' "${port}"
}

wait_for_server() {
    local host="$1"
    local port="$2"
    local pid="$3"
    local timeout="$4"

    local elapsed=0

    while (( elapsed < timeout )); do
        # Server 进程提前退出。
        if ! kill -0 "${pid}" 2>/dev/null; then
            return 2
        fi

        # 端口已经进入监听状态。
        if port_in_use "${host}" "${port}"; then
            return 0
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done

    return 1
}

# ============================================================================
# 子进程清理
# ============================================================================

kill_tree() {
    local parent_pid="$1"
    local signal="${2:-TERM}"

    local child_pids=""
    local child_pid=""

    child_pids="$(
        ps -o pid= --ppid "${parent_pid}" 2>/dev/null
    )" || true

    for child_pid in ${child_pids}; do
        kill_tree "${child_pid}" "${signal}"
    done

    kill -"${signal}" "${parent_pid}" 2>/dev/null || true
}

cleanup() {
    trap - EXIT INT TERM

    echo
    echo "[INFO] Cleaning evaluator and policy-server processes..."

    local pid=""

    # 先结束 evaluator。
    for pid in "${ACTIVE_EVAL_PIDS[@]-}"; do
        if [[ -n "${pid}" ]] &&
           kill -0 "${pid}" 2>/dev/null; then
            kill_tree "${pid}" TERM
        fi
    done

    # 再结束长期运行的 Policy Server。
    for pid in "${SLOT_SERVER_PIDS[@]-}"; do
        if [[ -n "${pid}" ]] &&
           kill -0 "${pid}" 2>/dev/null; then
            kill_tree "${pid}" TERM
        fi
    done

    sleep 2

    # TERM 无法结束时使用 KILL。
    for pid in \
        "${ACTIVE_EVAL_PIDS[@]-}" \
        "${SLOT_SERVER_PIDS[@]-}"; do

        if [[ -n "${pid}" ]] &&
           kill -0 "${pid}" 2>/dev/null; then
            kill_tree "${pid}" KILL
        fi
    done

    for pid in \
        "${ACTIVE_EVAL_PIDS[@]-}" \
        "${SLOT_SERVER_PIDS[@]-}"; do

        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup EXIT
trap 'exit 130' INT TERM

# ============================================================================
# 参数解析
# ============================================================================

parse_args() {
    while (( $# > 0 )); do
        case "$1" in
            -c|--ckpt)
                CKPT_PATH="${2:?Missing value for $1}"
                shift 2
                ;;

            -s|--slices)
                SLICES_PER_BENCHMARK="${2:?Missing value for $1}"
                shift 2
                ;;

            -p|--base-port)
                BASE_PORT="${2:?Missing value for $1}"
                shift 2
                ;;

            --host)
                HOST="${2:?Missing value for $1}"
                shift 2
                ;;

            --server-timeout)
                SERVER_TIMEOUT="${2:?Missing value for $1}"
                shift 2
                ;;

            --poll-interval)
                POLL_INTERVAL="${2:?Missing value for $1}"
                shift 2
                ;;

            --log-root)
                LOG_ROOT="${2:?Missing value for $1}"
                shift 2
                ;;

            --result-root)
                RESULT_ROOT="${2:?Missing value for $1}"
                shift 2
                ;;

            -h|--help)
                usage
                exit 0
                ;;

            -*)
                echo "[ERROR] Unknown option: $1" >&2
                usage >&2
                exit 1
                ;;

            *)
                REQUESTED_BENCHMARKS+=("$1")
                shift
                ;;
        esac
    done
}

resolve_benchmarks() {
    local -a resolved=()

    local requested=""
    local known=""
    local found=0

    if (( ${#REQUESTED_BENCHMARKS[@]} == 0 )); then
        REQUESTED_BENCHMARKS=("all")
    fi

    for requested in "${REQUESTED_BENCHMARKS[@]}"; do
        if [[ "${requested}" == "all" ]]; then
            REQUESTED_BENCHMARKS=("${BENCHMARKS[@]}")
            return 0
        fi

        found=0

        for known in "${BENCHMARKS[@]}"; do
            if [[ "${requested}" == "${known}" ]]; then
                found=1
                break
            fi
        done

        if (( found == 0 )); then
            echo "[ERROR] Unknown benchmark: ${requested}" >&2
            exit 1
        fi

        resolved+=("${requested}")
    done

    REQUESTED_BENCHMARKS=("${resolved[@]}")
}

# ============================================================================
# 构造切片任务队列
# ============================================================================

build_job_queue() {
    local task=""
    local size=0
    local base_size=0
    local remainder=0

    local start_idx=0
    local end_idx=0
    local slice_idx=0

    for task in "${REQUESTED_BENCHMARKS[@]}"; do
        size="$(benchmark_size "${task}")"

        base_size=$((size / SLICES_PER_BENCHMARK))
        remainder=$((size % SLICES_PER_BENCHMARK))

        start_idx=0

        for ((slice_idx = 0;
              slice_idx < SLICES_PER_BENCHMARK;
              ++slice_idx)); do

            if (( slice_idx < remainder )); then
                end_idx=$((start_idx + base_size + 1))
            else
                end_idx=$((start_idx + base_size))
            fi

            if (( slice_idx == SLICES_PER_BENCHMARK - 1 )); then
                end_idx="${size}"
            fi

            JOB_TASKS+=("${task}")
            JOB_STARTS+=("${start_idx}")
            JOB_ENDS+=("${end_idx}")
            JOB_SLICE_IDS+=("${slice_idx}")

            start_idx="${end_idx}"
        done
    done
}

# ============================================================================
# 为每张 GPU 启动一个长期 Policy Server
# ============================================================================

start_policy_servers() {
    mapfile -t SLOT_GPUS < <(detect_cuda_devices)

    local next_port="${BASE_PORT}"

    local slot_idx=0
    local gpu_id=""
    local port=""
    local server_log=""
    local server_pid=""
    local status=0

    # 先并行启动全部 Server。
    for slot_idx in "${!SLOT_GPUS[@]}"; do
        gpu_id="${SLOT_GPUS[$slot_idx]}"

        port="$(find_available_port "${next_port}")"
        next_port=$((port + 1))

        server_log="${
            RUN_LOG_DIR
        }/server_slot${slot_idx}_gpu${gpu_id}_port${port}.log"

        echo \
            "[SERVER] start slot=${slot_idx}" \
            "gpu=${gpu_id}" \
            "host=${HOST}" \
            "port=${port}"

        CUDA_VISIBLE_DEVICES="${gpu_id}" \
            bash "${SERVER_SCRIPT}" \
                "${CKPT_PATH}" \
                "${HOST}" \
                "${port}" \
            > "${server_log}" 2>&1 &

        server_pid=$!

        SLOT_PORTS[$slot_idx]="${port}"
        SLOT_SERVER_PIDS[$slot_idx]="${server_pid}"
        SLOT_SERVER_LOGS[$slot_idx]="${server_log}"
    done

    # 等待全部 Server 进入监听状态。
    for slot_idx in "${!SLOT_GPUS[@]}"; do
        port="${SLOT_PORTS[$slot_idx]}"
        server_pid="${SLOT_SERVER_PIDS[$slot_idx]}"
        server_log="${SLOT_SERVER_LOGS[$slot_idx]}"

        if wait_for_server \
            "${HOST}" \
            "${port}" \
            "${server_pid}" \
            "${SERVER_TIMEOUT}"; then

            echo "[SERVER] ready slot=${slot_idx} port=${port}"

        else
            status=$?

            if (( status == 2 )); then
                echo \
                    "[ERROR] Server exited before becoming ready:" \
                    "slot=${slot_idx}" >&2
            else
                echo \
                    "[ERROR] Server startup timeout:" \
                    "slot=${slot_idx}" >&2
            fi

            echo "[ERROR] Server log: ${server_log}" >&2

            tail -n 80 "${server_log}" >&2 || true

            exit 1
        fi
    done
}

# ============================================================================
# 在指定 slot 上启动一个切片
# ============================================================================

launch_job_in_slot() {
    local slot_idx="$1"
    local job_idx="$2"

    local gpu_id="${SLOT_GPUS[$slot_idx]}"
    local port="${SLOT_PORTS[$slot_idx]}"

    local task="${JOB_TASKS[$job_idx]}"
    local start_idx="${JOB_STARTS[$job_idx]}"
    local end_idx="${JOB_ENDS[$job_idx]}"
    local slice_id="${JOB_SLICE_IDS[$job_idx]}"

    local job_name="${
        task
    }_slice${slice_id}_${start_idx}_${end_idx}"

    local eval_log="${
        RUN_LOG_DIR
    }/${job_name}_slot${slot_idx}_gpu${gpu_id}.log"

    local result_dir="${
        RUN_RESULT_DIR
    }/${task}/${job_name}"

    mkdir -p "${result_dir}"

    echo \
        "[EVAL] start" \
        "slot=${slot_idx}" \
        "gpu=${gpu_id}" \
        "port=${port}" \
        "task=${task}" \
        "range=[${start_idx},${end_idx})"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    LIBERO_PRETRAINED_PATH="${CKPT_PATH}" \
        bash "${EVAL_SCRIPT}" \
            "${task}" \
            "${start_idx}" \
            "${end_idx}" \
            "${HOST}" \
            "${port}" \
            "${result_dir}" \
        > "${eval_log}" 2>&1 &

    ACTIVE_EVAL_PIDS[$slot_idx]="$!"
    ACTIVE_JOB_IDS[$slot_idx]="${job_idx}"
    ACTIVE_EVAL_LOGS[$slot_idx]="${eval_log}"
}

# ============================================================================
# 动态调度器
# ============================================================================

run_scheduler() {
    local total_jobs="${#JOB_TASKS[@]}"
    local total_slots="${#SLOT_GPUS[@]}"

    local next_job_idx=0
    local completed_jobs=0

    local slot_idx=0
    local eval_pid=""
    local server_pid=""

    local job_idx=""
    local task=""
    local slice_id=""
    local eval_log=""
    local exit_code=0

    while (( completed_jobs < total_jobs )); do
        for ((slot_idx = 0;
              slot_idx < total_slots;
              ++slot_idx)); do

            server_pid="${SLOT_SERVER_PIDS[$slot_idx]}"

            # 长期 Server 意外退出。
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                echo \
                    "[ERROR] Persistent policy server died:" \
                    "slot=${slot_idx}" >&2

                echo \
                    "[ERROR] Log:" \
                    "${SLOT_SERVER_LOGS[$slot_idx]}" >&2

                tail -n 80 \
                    "${SLOT_SERVER_LOGS[$slot_idx]}" \
                    >&2 || true

                exit 1
            fi

            eval_pid="${ACTIVE_EVAL_PIDS[$slot_idx]:-}"

            # 当前 evaluator 已经结束。
            if [[ -n "${eval_pid}" ]] &&
               ! kill -0 "${eval_pid}" 2>/dev/null; then

                job_idx="${ACTIVE_JOB_IDS[$slot_idx]}"
                task="${JOB_TASKS[$job_idx]}"
                slice_id="${JOB_SLICE_IDS[$job_idx]}"
                eval_log="${ACTIVE_EVAL_LOGS[$slot_idx]}"

                if wait "${eval_pid}"; then
                    echo \
                        "[EVAL] done" \
                        "slot=${slot_idx}" \
                        "task=${task}" \
                        "slice=${slice_id}"

                else
                    exit_code=$?

                    FAILED_JOBS+=(
                        "${task}:slice${slice_id}:exit${exit_code}"
                    )

                    echo \
                        "[EVAL] failed" \
                        "slot=${slot_idx}" \
                        "task=${task}" \
                        "slice=${slice_id}" \
                        "exit=${exit_code}" >&2

                    echo "[EVAL] log=${eval_log}" >&2
                fi

                ACTIVE_EVAL_PIDS[$slot_idx]=""
                ACTIVE_JOB_IDS[$slot_idx]=""
                ACTIVE_EVAL_LOGS[$slot_idx]=""

                completed_jobs=$((completed_jobs + 1))
            fi

            # 当前 slot 空闲，领取下一个任务。
            #
            # 注意：这里只重新启动 evaluator，
            # Policy Server 不会重新启动。
            if [[ -z "${ACTIVE_EVAL_PIDS[$slot_idx]:-}" ]] &&
               (( next_job_idx < total_jobs )); then

                launch_job_in_slot \
                    "${slot_idx}" \
                    "${next_job_idx}"

                next_job_idx=$((next_job_idx + 1))
            fi
        done

        if (( completed_jobs < total_jobs )); then
            sleep "${POLL_INTERVAL}"
        fi
    done
}

setup_environment() {
    # ============================================================
    # Project paths
    # ============================================================

    export STARVLA_DIR="/path/to/StarVLA"
    export LIBERO_HOME="/path/to/LIBERO"

    # ============================================================
    # Python environments
    # ============================================================

    export ABot_python="/path/to/starvla_env/bin/python"
    export LIBERO_PYTHON="/path/to/libero_env/bin/python"

    # ============================================================
    # Rendering
    # ============================================================

    export MUJOCO_GL="osmesa"

    # ============================================================
    # Model server
    # ============================================================

    export USE_BF16=1

    # ============================================================
    # HuggingFace / W&B
    # ============================================================

    export HF_ENDPOINT="https://hf-mirror.com"
    export WANDB_MODE="disabled"

    # ============================================================
    # Optional
    # ============================================================

    export TOKENIZERS_PARALLELISM="false"
}

main() {
    parse_args "$@"

    setup_environment

    if [[ -z "${CKPT_PATH}" ]]; then
        echo "[ERROR] --ckpt is required." >&2
        usage >&2
        exit 1
    fi

    if [[ ! -e "${CKPT_PATH}" ]]; then
        echo "[ERROR] Checkpoint not found: ${CKPT_PATH}" >&2
        exit 1
    fi

    # 转换为绝对路径，防止子脚本 cd 后相对路径失效。
    CKPT_PATH="$(
        cd "$(dirname "${CKPT_PATH}")" &&
        printf '%s/%s\n' "$PWD" "$(basename "${CKPT_PATH}")"
    )"

    if [[ ! -f "${SERVER_SCRIPT}" ]]; then
        echo "[ERROR] Server script not found: ${SERVER_SCRIPT}" >&2
        exit 1
    fi

    if [[ ! -f "${EVAL_SCRIPT}" ]]; then
        echo "[ERROR] Evaluator script not found: ${EVAL_SCRIPT}" >&2
        exit 1
    fi

    if ! [[ "${SLICES_PER_BENCHMARK}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] --slices must be a positive integer." >&2
        exit 1
    fi

    resolve_benchmarks

    RUN_LOG_DIR="${LOG_ROOT}/libero_eval_${RUN_ID}"
    RUN_RESULT_DIR="${RESULT_ROOT}/libero_eval_${RUN_ID}"

    mkdir -p \
        "${RUN_LOG_DIR}" \
        "${RUN_RESULT_DIR}"

    build_job_queue

    echo "============================================================"
    echo "[CONFIG] benchmarks : $(join_by ',' "${REQUESTED_BENCHMARKS[@]}")"
    echo "[CONFIG] slices     : ${SLICES_PER_BENCHMARK} per benchmark"
    echo "[CONFIG] jobs       : ${#JOB_TASKS[@]}"
    echo "[CONFIG] checkpoint : ${CKPT_PATH}"
    echo "[CONFIG] logs       : ${RUN_LOG_DIR}"
    echo "[CONFIG] results    : ${RUN_RESULT_DIR}"
    echo "============================================================"

    start_policy_servers

    echo "[CONFIG] GPUs        : $(join_by ',' "${SLOT_GPUS[@]}")"
    echo "[CONFIG] ports       : $(join_by ',' "${SLOT_PORTS[@]}")"
    echo "[INFO] Dynamic scheduling started."

    run_scheduler

    echo "============================================================"
    echo "[SUMMARY] total     : ${#JOB_TASKS[@]}"
    echo "[SUMMARY] succeeded : $((${#JOB_TASKS[@]} - ${#FAILED_JOBS[@]}))"
    echo "[SUMMARY] failed    : ${#FAILED_JOBS[@]}"
    echo "[SUMMARY] logs      : ${RUN_LOG_DIR}"
    echo "[SUMMARY] results   : ${RUN_RESULT_DIR}"
    echo "============================================================"

    if (( ${#FAILED_JOBS[@]} > 0 )); then
        printf \
            '[FAILED JOB] %s\n' \
            "${FAILED_JOBS[@]}" \
            >&2

        exit 1
    fi
}

main "$@"