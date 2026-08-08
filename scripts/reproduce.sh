#!/usr/bin/env bash
set -Eeuo pipefail

UV_RECOMMENDED_VERSION="0.11.29"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="$REPO_ROOT/results_replication"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$RESULTS_ROOT/reproduction/$RUN_ID"
LOG_FILE="$LOG_DIR/execution.log"
MANIFEST_FILE="$LOG_DIR/environment.json"
SUMMARY_FILE="$LOG_DIR/summary.json"
MAIN_PACKAGES_FILE="$LOG_DIR/main_packages.txt"
GPR_PACKAGES_FILE="$LOG_DIR/gpr_packages.txt"
START_EPOCH="$(date +%s)"
START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
UV_BIN=""
EXPERIMENT=""
MODE="resume"
MODE_EXPLICIT=0
PLOTS_ONLY=0
SKIP_WARMUP=0
CURRENT_STAGE="initialization"
COMPLETED_STAGES=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/reproduce.sh <experiment> [options]

Experiments:
  ablation      Run all ablation studies.
  validation    Validate the selected RFR default configuration.
  scalability   Run all scalability studies.
  comparison    Run the final classifier comparison.
  all           Run ablation, validation, scalability, and comparison.

Options:
  --clean        Delete replicated outputs for the selected experiment first.
  --resume       Resume from existing replicated CSV files (default).
  --plots-only   Regenerate plots without fitting estimators.
  --skip-warmup  Forward --skip-warmup to the Python experiment modules.
  -h, --help     Show this help message.
EOF
}

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
}

fail() {
    log "ERROR: $*"
    exit 1
}

write_summary() {
    local status="$1"
    local exit_code="$2"
    local end_epoch end_iso duration completed_json
    end_epoch="$(date +%s)"
    end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    duration=$((end_epoch - START_EPOCH))
    if ((${#COMPLETED_STAGES[@]})); then
        completed_json="$(printf '%s\n' "${COMPLETED_STAGES[@]}" | "$UV_BIN" run --project "$REPO_ROOT/environments/main" --locked python -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))' 2>/dev/null || printf '[]')"
    else
        completed_json='[]'
    fi
    cat > "$SUMMARY_FILE" <<EOF
{
  "status": "$status",
  "exit_code": $exit_code,
  "start_time_utc": "$START_ISO",
  "end_time_utc": "$end_iso",
  "duration_seconds": $duration,
  "selected_experiment": "$EXPERIMENT",
  "mode": "$MODE",
  "plots_only": $([[ "$PLOTS_ONLY" -eq 1 ]] && printf true || printf false),
  "skip_warmup": $([[ "$SKIP_WARMUP" -eq 1 ]] && printf true || printf false),
  "current_stage": "$CURRENT_STAGE",
  "completed_stages": $completed_json,
  "results_root": "${RESULTS_ROOT//\\/\\\\}"
}
EOF
}

on_exit() {
    local exit_code=$?
    trap - EXIT
    if [[ -n "$UV_BIN" && -d "$REPO_ROOT/environments/main" ]]; then
        if [[ $exit_code -eq 0 ]]; then
            write_summary "success" 0
        else
            write_summary "failed" "$exit_code"
        fi
    fi
    exit "$exit_code"
}
trap on_exit EXIT

parse_arguments() {
    [[ $# -gt 0 ]] || { usage; exit 2; }
    case "$1" in
        ablation|validation|scalability|comparison|all) EXPERIMENT="$1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown experiment: %s\n\n' "$1" >&2; usage; exit 2 ;;
    esac

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --clean)
                [[ "$MODE_EXPLICIT" -eq 0 ]] || fail "--clean and --resume cannot be used together."
                MODE="clean"; MODE_EXPLICIT=1
                ;;
            --resume)
                [[ "$MODE_EXPLICIT" -eq 0 ]] || fail "--clean and --resume cannot be used together."
                MODE="resume"; MODE_EXPLICIT=1
                ;;
            --plots-only) PLOTS_ONLY=1 ;;
            --skip-warmup) SKIP_WARMUP=1 ;;
            -h|--help) usage; exit 0 ;;
            *) printf 'Unknown option: %s\n\n' "$1" >&2; usage; exit 2 ;;
        esac
        shift
    done

    [[ ! ( "$MODE" == "clean" && "$PLOTS_ONLY" -eq 1 ) ]] || fail "--clean and --plots-only cannot be used together."
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return 0
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [[ -x "$candidate" ]]; then
            UV_BIN="$candidate"
            return 0
        fi
    done
    return 1
}

install_uv() {
    local url="https://astral.sh/uv/${UV_RECOMMENDED_VERSION}/install.sh"
    log "uv was not found. Installing recommended uv ${UV_RECOMMENDED_VERSION} with the official installer."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$url" | sh >>"$LOG_FILE" 2>&1
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$url" | sh >>"$LOG_FILE" 2>&1
    else
        cat >&2 <<'EOF'
curl or wget is required to install uv.
Install one of these tools and run the script again.

For example, on Ubuntu:
    sudo apt-get update
    sudo apt-get install curl ca-certificates
EOF
        exit 1
    fi
    find_uv || fail "uv installation completed, but the uv executable could not be found. Open a new shell and retry."
}

uv_version() {
    "$UV_BIN" --version | awk '{print $2}'
}

sync_environment() {
    local project="$1"
    CURRENT_STAGE="sync:$project"
    log "Synchronizing $project from its lockfile."
    if ! "$UV_BIN" sync --project "$REPO_ROOT/$project" --locked 2>&1 | tee -a "$LOG_FILE"; then
        log "Environment synchronization failed with uv $(uv_version)."
        log "The recommended uv version for these experiments is ${UV_RECOMMENDED_VERSION}."
        exit 1
    fi
}

python_path_for_project() {
    local project="$1"
    printf '%s/.venv/bin/python' "$REPO_ROOT/$project"
}

write_environment_manifest() {
    local os_name os_version architecture hostname_value cpu_count memory_bytes main_python gpr_python detected_uv
    os_name="$(uname -s 2>/dev/null || printf unknown)"
    os_version="$(uname -r 2>/dev/null || printf unknown)"
    architecture="$(uname -m 2>/dev/null || printf unknown)"
    hostname_value="$(hostname 2>/dev/null || printf unknown)"
    cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || printf unknown)"
    memory_bytes="unknown"
    if [[ -r /proc/meminfo ]]; then
        memory_bytes="$(( $(awk '/MemTotal:/ {print $2}' /proc/meminfo) * 1024 ))"
    fi
    detected_uv="$($UV_BIN --version | tr -d '\r')"
    main_python="$($UV_BIN run --project "$REPO_ROOT/environments/main" --locked python --version 2>&1 | tr -d '\r')"
    gpr_python="$($UV_BIN run --project "$REPO_ROOT/environments/gpr" --locked python --version 2>&1 | tr -d '\r')"

    cat > "$MANIFEST_FILE" <<EOF
{
  "start_time_utc": "$START_ISO",
  "operating_system": "$os_name",
  "operating_system_version": "$os_version",
  "architecture": "$architecture",
  "hostname": "$hostname_value",
  "logical_cpu_count": "$cpu_count",
  "total_memory_bytes": "$memory_bytes",
  "uv_version": "$detected_uv",
  "recommended_uv_version": "$UV_RECOMMENDED_VERSION",
  "main_python_version": "$main_python",
  "gpr_python_version": "$gpr_python",
  "selected_experiment": "$EXPERIMENT",
  "mode": "$MODE",
  "plots_only": $([[ "$PLOTS_ONLY" -eq 1 ]] && printf true || printf false),
  "skip_warmup": $([[ "$SKIP_WARMUP" -eq 1 ]] && printf true || printf false),
  "results_root": "${RESULTS_ROOT//\\/\\\\}"
}
EOF

    "$UV_BIN" pip freeze --python "$(python_path_for_project environments/main)" > "$MAIN_PACKAGES_FILE"
    "$UV_BIN" pip freeze --python "$(python_path_for_project environments/gpr)" > "$GPR_PACKAGES_FILE"
}

smoke_test() {
    CURRENT_STAGE="smoke-test"
    log "Running import smoke tests."
    "$UV_BIN" run --project "$REPO_ROOT/environments/main" --locked python -c \
        "import numpy, pandas, sklearn, numba, aeon, imodels; from random_fuzzy_rules import RandomFuzzyRulesClassifier; import experiments.utils; print('Main environment smoke test passed.')" \
        2>&1 | tee -a "$LOG_FILE"
    "$UV_BIN" run --project "$REPO_ROOT/environments/gpr" --locked python -c \
        "import geppy, deap; from gpr_fast import GPR_FAST; print('GPR environment smoke test passed.')" \
        2>&1 | tee -a "$LOG_FILE"
}

safe_remove() {
    local target="$1" canonical_root canonical_target
    [[ -e "$target" ]] || return 0
    canonical_root="$(cd "$RESULTS_ROOT" && pwd -P)"
    canonical_target="$(cd "$(dirname "$target")" && pwd -P)/$(basename "$target")"
    case "$canonical_target" in
        "$canonical_root"/*) ;;
        *) fail "Refusing to remove path outside results_replication: $canonical_target" ;;
    esac
    [[ "$canonical_target" != "$canonical_root" ]] || fail "Refusing to remove the results_replication root."
    log "Removing replicated output: $canonical_target"
    rm -rf -- "$canonical_target"
}

clean_selected_outputs() {
    [[ "$MODE" == "clean" ]] || return 0
    case "$EXPERIMENT" in
        ablation) safe_remove "$RESULTS_ROOT/ablation" ;;
        validation) safe_remove "$RESULTS_ROOT/ablation/default_configuration_validation" ;;
        scalability) safe_remove "$RESULTS_ROOT/scalability" ;;
        comparison) safe_remove "$RESULTS_ROOT/comparison" ;;
        all)
            safe_remove "$RESULTS_ROOT/ablation"
            safe_remove "$RESULTS_ROOT/scalability"
            safe_remove "$RESULTS_ROOT/comparison"
            ;;
    esac
}

run_python_module() {
    local stage="$1" module="$2"
    shift 2
    CURRENT_STAGE="$stage"
    log "Starting experiment stage: $stage"
    "$UV_BIN" run --project "$REPO_ROOT/environments/main" --locked python -m "$module" "$@" 2>&1 | tee -a "$LOG_FILE"
    COMPLETED_STAGES+=("$stage")
    log "Completed experiment stage: $stage"
}

run_stage() {
    local stage="$1"
    local common=(--results-root "$RESULTS_ROOT")
    [[ "$PLOTS_ONLY" -eq 0 ]] || common+=(--plots-only)
    [[ "$SKIP_WARMUP" -eq 0 ]] || common+=(--skip-warmup)
    case "$stage" in
        ablation) run_python_module ablation experiments.ablation.run --study all "${common[@]}" ;;
        validation) run_python_module validation experiments.ablation.validate_default_configuration "${common[@]}" ;;
        scalability) run_python_module scalability experiments.scalability.run --study all "${common[@]}" ;;
        comparison) run_python_module comparison experiments.comparison.run "${common[@]}" ;;
    esac
}

main() {
    parse_arguments "$@"
    mkdir -p "$LOG_DIR"
    touch "$LOG_FILE"
    cd "$REPO_ROOT"

    if [[ "$MODE_EXPLICIT" -eq 0 ]]; then
        log "Neither --clean nor --resume was specified. Resume mode is used by default."
        log "Only files under $RESULTS_ROOT are reused. The reference results under $REPO_ROOT/results are not modified."
    fi

    if ! find_uv; then
        install_uv
    fi
    local detected_version
    detected_version="$(uv_version)"
    log "Using uv $detected_version at $UV_BIN"
    if [[ "$detected_version" != "$UV_RECOMMENDED_VERSION" ]]; then
        log "WARNING: uv $detected_version differs from the recommended version $UV_RECOMMENDED_VERSION. Continuing with the installed version."
    fi

    sync_environment environments/main
    sync_environment environments/gpr
    write_environment_manifest
    smoke_test
    clean_selected_outputs

    if [[ "$EXPERIMENT" == "all" ]]; then
        run_stage ablation
        run_stage validation
        run_stage scalability
        run_stage comparison
    else
        run_stage "$EXPERIMENT"
    fi
    CURRENT_STAGE="completed"
}

main "$@"
