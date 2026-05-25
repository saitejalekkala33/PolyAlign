#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/umair/TW/PolyAlign}
DATA_ROOT=${DATA_ROOT:-$REPO/data/chinese/merged_sft_dedup}
RUNS_DIR=${RUNS_DIR:-$DATA_ROOT/runs}
METRICS_DIR=${METRICS_DIR:-$REPO/data/metrics}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-$REPO/logs/hdpo_metrics_zh/$RUN_ID}
PREP_DIR=${PREP_DIR:-$LOG_DIR/prepared_inputs}
GIT_LD_LIBRARY_PATH=${GIT_LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu}
GIT_LOCK_PATH=${GIT_LOCK_PATH:-$LOG_DIR/git-push.lock}
OVERWRITE_ARTIFACTS=${OVERWRITE_ARTIFACTS:-1}
RUN_SUFFIX=${RUN_SUFFIX:-}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${MODELS:-}" ]]; then
  read -r -a models <<<"$MODELS"
else
  models=(llama32_3b qwen25_1_5b gemma2_2b qwen25_3b)
fi

declare -A run_name gpu
run_name[llama32_3b]=llama32-3b-hdpo-zh
run_name[qwen25_1_5b]=qwen25-1-5b-hdpo-zh
run_name[gemma2_2b]=gemma2-2b-hdpo-zh
run_name[qwen25_3b]=qwen25-3b-hdpo-zh

if [[ -n "$RUN_SUFFIX" ]]; then
  for alias in llama32_3b qwen25_1_5b gemma2_2b qwen25_3b; do
    run_name[$alias]="${run_name[$alias]}$RUN_SUFFIX"
  done
fi

gpu[llama32_3b]=4
gpu[qwen25_1_5b]=5
gpu[gemma2_2b]=6
gpu[qwen25_3b]=7

git_cmd() {
  LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

commit_metric_json() {
  local metric_path="$1"
  local message="$2"

  git_cmd pull --ff-only
  git_cmd add "$metric_path"
  if ! git_cmd diff --cached --quiet -- "$metric_path"; then
    git_cmd commit -m "$message"
    git_cmd push
  else
    git_cmd reset --quiet -- "$metric_path"
    echo "No metric JSON changes to commit for $metric_path"
  fi
}

commit_metric_json_with_lock() {
  local metric_path="$1"
  local message="$2"

  if command -v flock >/dev/null 2>&1; then
    (
      flock 200
      commit_metric_json "$metric_path" "$message"
    ) 200>"$GIT_LOCK_PATH"
    return
  fi

  local lock_dir="$GIT_LOCK_PATH.dir"
  while ! mkdir "$lock_dir" 2>/dev/null; do
    sleep 5
  done
  trap 'rmdir "$lock_dir" 2>/dev/null || true' RETURN
  commit_metric_json "$metric_path" "$message"
  rmdir "$lock_dir" 2>/dev/null || true
  trap - RETURN
}

prepare_metric_inputs() {
  local alias="$1"
  local hdpo_test_path="$DATA_ROOT/hdpo_prepared/$alias/llamafactory/hdpo_test.json"
  local current_hdpo_path="$DATA_ROOT/current-hdpo-zh/$alias/current_hdpo_test.jsonl"
  local out_dir="$PREP_DIR/$alias"
  local out_test_path="$out_dir/test.json"
  local out_current_path="$out_dir/current.jsonl"
  local out_report_path="$out_dir/alignment_report.json"

  require_file "$hdpo_test_path"
  require_file "$current_hdpo_path"
  mkdir -p "$out_dir"

  python - "$hdpo_test_path" "$current_hdpo_path" "$out_test_path" "$out_current_path" "$out_report_path" <<'PY'
import json
import sys
from collections import defaultdict, deque
from pathlib import Path


def norm(value):
    if value is None:
        return ""
    return "\n".join(" ".join(line.strip().split()) for line in str(value).replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")).strip()


def lf_history(row):
    history = row.get("history") or []
    if not isinstance(history, list):
        return []
    pairs = []
    for turn in history:
        if isinstance(turn, list) and len(turn) == 2:
            user = norm(turn[0])
            assistant = norm(turn[1])
            if user or assistant:
                pairs.append((user, assistant))
    return tuple(pairs)


def current_history(row):
    turns = row.get("dialogue_history") or []
    if not isinstance(turns, list):
        return tuple()
    pairs = []
    pending_user = None
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = norm(turn.get("role"))
        text = norm(turn.get("text", turn.get("content", "")))
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, text))
            pending_user = None
    return tuple(pairs)


def chosen(row):
    return norm(row.get("chosen") or row.get("reference_output") or row.get("output") or row.get("human_answer"))


def lf_exact_key(row):
    return (norm(row.get("instruction")), norm(row.get("input")), chosen(row), lf_history(row))


def lf_prompt_key(row):
    return (norm(row.get("instruction")), norm(row.get("input")), lf_history(row))


def current_exact_key(row):
    return (norm(row.get("question")), norm(row.get("context")), norm(row.get("human_answer")), current_history(row))


def current_prompt_key(row):
    return (norm(row.get("question")), norm(row.get("context")), current_history(row))


def pop_unused(queue, used):
    while queue:
        index, row = queue.popleft()
        if index not in used:
            used.add(index)
            return index, row
    return None, None


hdpo_path, current_path, out_test, out_current, out_report = map(Path, sys.argv[1:])
hdpo_rows = json.loads(hdpo_path.read_text(encoding="utf-8"))
current_rows = [json.loads(line) for line in current_path.read_text(encoding="utf-8").splitlines() if line.strip()]

exact_map = defaultdict(deque)
prompt_map = defaultdict(deque)
for idx, row in enumerate(current_rows):
    exact_map[current_exact_key(row)].append((idx, row))
    prompt_map[current_prompt_key(row)].append((idx, row))

used = set()
metric_test_rows = []
metric_current_rows = []
counts = {
    "position_exact": 0,
    "signature_exact": 0,
    "prompt_fallback": 0,
    "missing": 0,
}

for idx, row in enumerate(hdpo_rows):
    ref = chosen(row)
    if not ref:
        raise ValueError(f"HDPO row {idx} has no chosen/reference output.")

    matched = None
    if idx < len(current_rows) and idx not in used and current_exact_key(current_rows[idx]) == lf_exact_key(row):
        matched = dict(current_rows[idx])
        used.add(idx)
        counts["position_exact"] += 1

    if matched is None:
        match_index, match_row = pop_unused(exact_map[lf_exact_key(row)], used)
        if match_row is not None:
            matched = dict(match_row)
            counts["signature_exact"] += 1

    if matched is None:
        match_index, match_row = pop_unused(prompt_map[lf_prompt_key(row)], used)
        if match_row is not None:
            matched = dict(match_row)
            counts["prompt_fallback"] += 1

    if matched is None:
        counts["missing"] += 1
        raise ValueError(
            "Could not align HDPO row to current row: "
            f"index={idx}, instruction={norm(row.get('instruction'))[:120]!r}"
        )

    metric_row = dict(row)
    metric_row["output"] = ref
    metric_row["instruction"] = norm(metric_row.get("instruction"))
    metric_row["input"] = norm(metric_row.get("input"))
    if "history" in metric_row:
        metric_row["history"] = [[user, assistant] for user, assistant in lf_history(row)]
    metric_test_rows.append(metric_row)

    matched["question"] = metric_row["instruction"]
    matched["context"] = metric_row["input"]
    matched["human_answer"] = ref
    matched["language"] = matched.get("language") or metric_row.get("language", "")
    matched["track"] = matched.get("track") or metric_row.get("track", "")
    matched["family"] = matched.get("family") or metric_row.get("family", "")
    matched["style_bucket"] = matched.get("style_bucket") or metric_row.get("style_bucket", "")
    matched["length_bin"] = matched.get("length_bin") or metric_row.get("length_bin", "")
    matched["bucket_id"] = matched.get("bucket_id") or metric_row.get("bucket_id", "")
    metric_current_rows.append(matched)

out_test.parent.mkdir(parents=True, exist_ok=True)
out_test.write_text(json.dumps(metric_test_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
with out_current.open("w", encoding="utf-8") as handle:
    for row in metric_current_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

report = {
    "hdpo_rows": len(hdpo_rows),
    "current_rows": len(current_rows),
    "metric_rows": len(metric_test_rows),
    "used_current_rows": len(used),
    **counts,
}
out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
PY

  echo "$out_test_path|$out_current_path|$out_report_path"
}

run_metrics_for_model() {
  local alias="$1"
  local name="${run_name[$alias]}"
  local gpu_id="${gpu[$alias]}"
  local run_dir="$RUNS_DIR/$name"
  local predictions_path="$run_dir/predictions.jsonl"
  local human_feature_path="$DATA_ROOT/features-hdpo/research_models/test/$alias/test_answer_features_dedup.jsonl"
  local bucket_references_path="$DATA_ROOT/reference_artifacts-hdpo/$alias/bucket_references.json"
  local feature_matrix_path="$DATA_ROOT/reference_artifacts-hdpo/$alias/feature_matrix.jsonl"
  local metric_json="$METRICS_DIR/${name}-test.json"
  local metric_work_dir="$METRICS_DIR/${name}-artifacts"
  local job_log="$LOG_DIR/$name.metrics-only.job.log"
  local metrics_log="$LOG_DIR/$name.metrics.log"
  local git_log="$LOG_DIR/$name.git.log"

  (
    set -euo pipefail
    require_file "$predictions_path"
    require_file "$human_feature_path"
    require_file "$bucket_references_path"
    require_file "$feature_matrix_path"

    local prepared
    prepared="$(prepare_metric_inputs "$alias")"
    local test_lf_path="${prepared%%|*}"
    local rest="${prepared#*|}"
    local current_test_path="${rest%%|*}"
    local report_path="${rest#*|}"
    echo "Prepared metric inputs for $name: $report_path"

    local overwrite_args=()
    if [[ "$OVERWRITE_ARTIFACTS" == "1" ]]; then
      overwrite_args=(--overwrite-artifacts)
    fi

    echo "=== Metrics: $name on CUDA device $gpu_id ==="
    CUDA_VISIBLE_DEVICES="$gpu_id" python -m metrics \
      --test-lf-path "$test_lf_path" \
      --predictions-path "$predictions_path" \
      --output-json "$metric_json" \
      --current-test-path "$current_test_path" \
      --human-feature-path "$human_feature_path" \
      --bucket-references-path "$bucket_references_path" \
      --feature-matrix-path "$feature_matrix_path" \
      --work-dir "$metric_work_dir" \
      --model-alias "$alias" \
      --device cuda \
      --mauve-device-id 0 \
      "${overwrite_args[@]}" \
      2>&1 | tee "$metrics_log"

    echo "Committing metrics for $name"
    commit_metric_json_with_lock "$metric_json" "add chinese hdpo metrics for $name" 2>&1 | tee "$git_log"
    echo "=== Completed metrics for $name ==="
  ) > >(tee "$job_log") 2>&1
}

mkdir -p "$LOG_DIR" "$PREP_DIR" "$METRICS_DIR"
echo "Logs directory: $LOG_DIR"

declare -A job_pids

cleanup_jobs() {
  for pid in "${job_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup_jobs INT TERM

for alias in "${models[@]}"; do
  run_metrics_for_model "$alias" &
  job_pids[$alias]=$!
  echo "Started metrics for ${run_name[$alias]} on CUDA device ${gpu[$alias]} with PID ${job_pids[$alias]}"
done

status=0
for alias in "${models[@]}"; do
  if wait "${job_pids[$alias]}"; then
    echo "Finished metrics for ${run_name[$alias]}"
  else
    echo "Failed metrics for ${run_name[$alias]} (see $LOG_DIR/${run_name[$alias]}.metrics-only.job.log)" >&2
    status=1
  fi
done

exit "$status"
