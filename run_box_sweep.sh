#!/usr/bin/env bash
# 2-shard NUMA-pinned throughput sweep across three modes (offline, single, batch=48).
# Each pass launches one speech_benchmark.py per socket, waits for both, then
# aggregates the freshly-written per-shard CSVs into output/pytorch/box_summary.csv.
#
# Hardware target: dual Xeon 8568Y+ (2 sockets × 48 physical cores). One process
# per NUMA node, 48 OpenMP threads each — no HT siblings, no cross-socket UPI.
#
# Re-run safely: each pass tags its logs with a per-pass timestamp, and the
# aggregator picks the NEWEST matching pair via `ls -t | head -1`, so old shard
# CSVs in output/pytorch/ never get accidentally re-aggregated.

set -eo pipefail

cd "$(dirname "$(readlink -f "$0")")"

OUT_DIR=output/pytorch
mkdir -p "$OUT_DIR"

MODEL=whisper-large-v3
DTYPE=bf16
DATASET=librispeech
THREADS=48

# Run one sharded pass (shard 0 on socket 0, shard 1 on socket 1), then aggregate.
#   $1 = label   — short name used in log filenames and section header
#   $2 = tag     — speech_benchmark.py's filename suffix for this mode
#                  (e.g. offline-bAll, single-b1, batch-b48)
#   $@ remaining — extra args forwarded to speech_benchmark.py
run_sharded() {
    local label="$1"; shift
    local tag="$1";   shift
    local extra=("$@")

    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    local log0="$OUT_DIR/${label}_sock0_${stamp}.log"
    local log1="$OUT_DIR/${label}_sock1_${stamp}.log"

    echo
    echo "================================================================"
    echo "Sweep: $label   ($(date +%H:%M:%S))"
    echo "================================================================"

    numactl --cpunodebind=0 --membind=0 \
        env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
        python speech_benchmark.py --model "$MODEL" --dtype "$DTYPE" \
            --datasets "$DATASET" "${extra[@]}" \
            --num-shards 2 --shard-idx 0 \
            > "$log0" 2>&1 &
    local pid0=$!

    numactl --cpunodebind=1 --membind=1 \
        env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
        python speech_benchmark.py --model "$MODEL" --dtype "$DTYPE" \
            --datasets "$DATASET" "${extra[@]}" \
            --num-shards 2 --shard-idx 1 \
            > "$log1" 2>&1 &
    local pid1=$!

    echo "shard 0: pid $pid0   log $log0"
    echo "shard 1: pid $pid1   log $log1"
    echo "waiting…"

    # Don't let set -e bail on first non-zero. We want to surface both exit codes.
    set +e
    wait $pid0; local rc0=$?
    wait $pid1; local rc1=$?
    set -e
    echo "shard 0 exit $rc0,  shard 1 exit $rc1"

    if [[ $rc0 -ne 0 && $rc0 -ne 134 ]]; then
        echo "[FAIL] shard 0 failed for $label — see $log0" >&2
        return 1
    fi
    if [[ $rc1 -ne 0 && $rc1 -ne 134 ]]; then
        echo "[FAIL] shard 1 failed for $label — see $log1" >&2
        return 1
    fi
    # rc=134 (SIGABRT during finalizer teardown) is the known torch/CPU
    # cleanup race — the CSV is already written by then.

    # Pick up the newest shard CSV for this tag (the run we just did).
    local shard0_csv shard1_csv
    shard0_csv=$(ls -t "$OUT_DIR"/*"_${DATASET}_${tag}-shard0of2_"*.csv 2>/dev/null | head -1)
    shard1_csv=$(ls -t "$OUT_DIR"/*"_${DATASET}_${tag}-shard1of2_"*.csv 2>/dev/null | head -1)
    if [[ -z "$shard0_csv" || -z "$shard1_csv" ]]; then
        echo "[FAIL] could not locate per-shard CSVs for tag '$tag'" >&2
        return 1
    fi
    echo "aggregating:"
    echo "  $shard0_csv"
    echo "  $shard1_csv"

    python aggregate_shards.py "$shard0_csv" "$shard1_csv" \
        --write-csv "$OUT_DIR/box_summary.csv"
}


# === The sweep ===

run_sharded "offline"  "offline-bAll" --mode offline
run_sharded "single"   "single-b1"    --mode single
run_sharded "batch48"  "batch-b48"    --mode batch --batch-size 48


echo
echo "================================================================"
echo "DONE  ($(date +%H:%M:%S))"
echo "Box-level summary rows appended to: $OUT_DIR/box_summary.csv"
echo "================================================================"
column -s, -t < "$OUT_DIR/box_summary.csv" | tail -4
