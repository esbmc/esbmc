#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh  —  ESBMC-PLC vs NuXmv BDD comparison experiment runner
#
# For each benchmark:
#   1. Run ESBMC-PLC (k-induction + incremental BMC)
#   2. Transpile LD → SMV via ld_to_smv.py
#   3. Run NuXmv in BDD mode (LTLSPEC via BDD)
#   4. Run NuXmv in IC3 mode (best SAT-based)
#   5. Record verdict and wall-clock time for each
#
# Output: results/results.csv and results/results_table.tex
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_DIR="${BENCH_DIR:-$ARTIFACT_DIR/benchmarks}"
ESBMC="${ESBMC:-$(command -v esbmc 2>/dev/null || echo "")}"
NUXMV="${NUXMV:-$(command -v nuXmv 2>/dev/null || command -v nuxmv 2>/dev/null || echo "")}"
TRANSPILER="$SCRIPT_DIR/ld_to_smv.py"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMEOUT=120   # seconds per run

mkdir -p "$RESULTS_DIR"
SMV_DIR="$RESULTS_DIR/smv"
mkdir -p "$SMV_DIR"

CSV="$RESULTS_DIR/results.csv"
echo "benchmark,ld_file,expected,esbmc_verdict,esbmc_time_s,nuxmv_bdd_verdict,nuxmv_bdd_time_s,nuxmv_ic3_verdict,nuxmv_ic3_time_s,num_bool_vars,num_int_vars,num_rungs,num_props" > "$CSV"

# ---------------------------------------------------------------------------
# Benchmark manifest: (label, ld_file, props_file, expected_verdict)
# ---------------------------------------------------------------------------
declare -a BENCHMARKS=(
    # Boolean-only programs (no integer state)
    "tank_level_safe|$BENCH_DIR/tank_level_control/tank_level_control.ld|$BENCH_DIR/tank_level_control/props.yaml|SAFE"
    "tank_level_unsafe|$BENCH_DIR/tank_level_control/tank_level_control_unsafe.ld|$BENCH_DIR/tank_level_control/props.yaml|VIOLATION"

    # Timer programs (INT state variables — BDD stress test)
    "bottle_filling_safe|$BENCH_DIR/bottle_filling/bottle_filling_safe.ld|$BENCH_DIR/bottle_filling/props.yaml|SAFE"
    "bottle_filling_unsafe|$BENCH_DIR/bottle_filling/bottle_filling_unsafe.ld|$BENCH_DIR/bottle_filling/props.yaml|VIOLATION"
    "elevator_safe|$BENCH_DIR/elevator/elevator_safe.ld|$BENCH_DIR/elevator/props.yaml|SAFE"
    "elevator_unsafe|$BENCH_DIR/elevator/elevator_unsafe.ld|$BENCH_DIR/elevator/props_unsafe.yaml|VIOLATION"
    "traffic_light_safe|$BENCH_DIR/traffic_light/traffic_light_safe.ld|$BENCH_DIR/traffic_light/props.yaml|SAFE"
    "traffic_light_unsafe|$BENCH_DIR/traffic_light/traffic_light_unsafe.ld|$BENCH_DIR/traffic_light/props.yaml|VIOLATION"
)

# ---------------------------------------------------------------------------
# Helper: run with timeout, return wall-clock time and exit code
# ---------------------------------------------------------------------------
run_timed() {
    local out_file="$1"; shift
    local start end elapsed rc=0
    start=$(python3 -c 'import time; print(time.time())')
    timeout "$TIMEOUT" "$@" > "$out_file" 2>&1 || rc=$?
    end=$(python3 -c 'import time; print(time.time())')
    elapsed=$(python3 -c "print(f'{$end - $start:.3f}')")
    echo "$elapsed $rc"
}

# ---------------------------------------------------------------------------
# Helper: parse ESBMC verdict
# ---------------------------------------------------------------------------
esbmc_verdict() {
    local log="$1"
    if grep -q "VERIFICATION SUCCESSFUL" "$log" 2>/dev/null; then
        echo "SAFE"
    elif grep -q "VERIFICATION FAILED" "$log" 2>/dev/null; then
        echo "VIOLATION"
    else
        echo "UNKNOWN"
    fi
}

# ---------------------------------------------------------------------------
# Helper: parse NuXmv verdict from LTLSPEC output
# Specification is true  → SAFE (property holds)
# Specification is false → VIOLATION (counterexample exists)
# ---------------------------------------------------------------------------
nuxmv_verdict() {
    local log="$1" rc="$2"
    if [ "$rc" -eq 124 ]; then
        echo "TIMEOUT"
        return
    fi
    if grep -q "is true" "$log" 2>/dev/null; then
        # All specs true → SAFE only if no false spec exists
        if grep -q "is false" "$log" 2>/dev/null; then
            echo "VIOLATION"
        else
            echo "SAFE"
        fi
    elif grep -q "is false" "$log" 2>/dev/null; then
        echo "VIOLATION"
    else
        echo "UNKNOWN"
    fi
}

# ---------------------------------------------------------------------------
# Helper: count variable types in LD file
# ---------------------------------------------------------------------------
count_vars() {
    local ld="$1"
    local bool_cnt int_cnt rung_cnt
    bool_cnt=$(grep -c '<BOOL/>' "$ld" 2>/dev/null || echo 0)
    int_cnt=$(grep -cE '<(INT|DINT|UINT|WORD|BYTE)\/>' "$ld" 2>/dev/null || echo 0)
    rung_cnt=$(grep -c '<rung' "$ld" 2>/dev/null || echo 0)
    echo "$bool_cnt $int_cnt $rung_cnt"
}

# ---------------------------------------------------------------------------
# NuXmv BDD command script
# ---------------------------------------------------------------------------
nuxmv_bdd_cmds() {
    local smv="$1"
    cat <<EOF
read_model -i $smv
flatten_hierarchy
encode_variables
build_model
check_ltlspec
quit
EOF
}

# NuXmv IC3 command script
nuxmv_ic3_cmds() {
    local smv="$1"
    cat <<EOF
read_model -i $smv
flatten_hierarchy
encode_variables
build_boolean_model
check_ltlspec_ic3
quit
EOF
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo "========================================================"
echo "  ESBMC-PLC vs NuXmv BDD/IC3 — Experiment Runner"
echo "  $(date)"
echo "========================================================"
echo ""

for entry in "${BENCHMARKS[@]}"; do
    IFS='|' read -r label ld_file props_file expected <<< "$entry"

    echo "──────────────────────────────────────────────────────"
    echo "  Benchmark: $label  (expected: $expected)"

    # Skip missing files
    if [ ! -f "$ld_file" ]; then
        echo "  [SKIP] LD file not found: $ld_file"
        continue
    fi

    # Count vars
    read -r bool_cnt int_cnt rung_cnt <<< "$(count_vars "$ld_file")"

    # Count properties
    prop_cnt=$(python3 -c "
import yaml
with open('$props_file') as f:
    d = yaml.safe_load(f)
print(len(d.get('properties',[])))
" 2>/dev/null || echo 0)

    echo "  Vars: ${bool_cnt} BOOL, ${int_cnt} INT | Rungs: $rung_cnt | Props: $prop_cnt"

    # ------------------------------------------------------------------
    # 1. Transpile LD → SMV
    # ------------------------------------------------------------------
    smv_file="$SMV_DIR/${label}.smv"
    if python3 "$TRANSPILER" "$ld_file" "$props_file" --out "$smv_file" 2>/dev/null; then
        echo "  [SMV] Generated: $smv_file"
    else
        echo "  [SMV] Transpilation failed — skipping NuXmv runs"
        smv_file=""
    fi

    # ------------------------------------------------------------------
    # 2. ESBMC-PLC (k-induction for SAFE, incremental BMC for violations)
    # ------------------------------------------------------------------
    esbmc_log="$RESULTS_DIR/${label}_esbmc.log"
    echo -n "  [ESBMC] Running ... "
    read -r esbmc_time esbmc_rc <<< "$(run_timed "$esbmc_log" \
        "$ESBMC" "$ld_file" --ld-props "$props_file" \
        --k-induction --z3 --no-div-by-zero-check 2>&1 || true)"
    ESBMC_VERDICT=$(esbmc_verdict "$esbmc_log")
    echo "${ESBMC_VERDICT} in ${esbmc_time}s"

    # ------------------------------------------------------------------
    # 3. NuXmv BDD
    # ------------------------------------------------------------------
    nuxmv_bdd_time="N/A"
    nuxmv_bdd_verdict="N/A"
    if [ -n "$smv_file" ] && [ -f "$smv_file" ]; then
        bdd_log="$RESULTS_DIR/${label}_nuxmv_bdd.log"
        echo -n "  [NuXmv BDD] Running ... "
        bdd_cmds_file=$(mktemp /tmp/nuxmv_bdd_XXXXXX)
        nuxmv_bdd_cmds "$smv_file" > "$bdd_cmds_file"
        bdd_start=$(python3 -c 'import time; print(time.time())')
        bdd_rc=0; timeout "$TIMEOUT" "$NUXMV" -int < "$bdd_cmds_file" > "$bdd_log" 2>&1 || bdd_rc=$?
        bdd_end=$(python3 -c 'import time; print(time.time())')
        nuxmv_bdd_time=$(python3 -c "print(f'{$bdd_end - $bdd_start:.3f}')")
        rm -f "$bdd_cmds_file"
        nuxmv_bdd_verdict=$(nuxmv_verdict "$bdd_log" "$bdd_rc")
        echo "${nuxmv_bdd_verdict} in ${nuxmv_bdd_time}s"
    fi

    # ------------------------------------------------------------------
    # 4. NuXmv IC3
    # ------------------------------------------------------------------
    nuxmv_ic3_time="N/A"
    nuxmv_ic3_verdict="N/A"
    if [ -n "$smv_file" ] && [ -f "$smv_file" ]; then
        ic3_log="$RESULTS_DIR/${label}_nuxmv_ic3.log"
        echo -n "  [NuXmv IC3] Running ... "
        ic3_cmds_file=$(mktemp /tmp/nuxmv_ic3_XXXXXX)
        nuxmv_ic3_cmds "$smv_file" > "$ic3_cmds_file"
        ic3_start=$(python3 -c 'import time; print(time.time())')
        ic3_rc=0; timeout "$TIMEOUT" "$NUXMV" -int < "$ic3_cmds_file" > "$ic3_log" 2>&1 || ic3_rc=$?
        ic3_end=$(python3 -c 'import time; print(time.time())')
        nuxmv_ic3_time=$(python3 -c "print(f'{$ic3_end - $ic3_start:.3f}')")
        rm -f "$ic3_cmds_file"
        nuxmv_ic3_verdict=$(nuxmv_verdict "$ic3_log" "$ic3_rc")
        echo "${nuxmv_ic3_verdict} in ${nuxmv_ic3_time}s"
    fi

    echo "$label,$ld_file,$expected,$ESBMC_VERDICT,$esbmc_time,$nuxmv_bdd_verdict,$nuxmv_bdd_time,$nuxmv_ic3_verdict,$nuxmv_ic3_time,$bool_cnt,$int_cnt,$rung_cnt,$prop_cnt" >> "$CSV"
    echo ""
done

echo "========================================================"
echo "  Results written to: $CSV"
echo "========================================================"

# Generate LaTeX table
python3 "$SCRIPT_DIR/make_table.py" "$CSV" > "$RESULTS_DIR/results_table.tex"
echo "  LaTeX table: $RESULTS_DIR/results_table.tex"
