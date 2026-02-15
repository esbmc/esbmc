#!/bin/bash
#
# full-audit.sh - Comprehensive ESBMC security audit
#
# Usage: ./full-audit.sh <file> [options]
#
# Performs a thorough security audit using multiple verification passes
# with different configurations to maximize bug detection.
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default settings
TIMEOUT="120s"
MAX_UNWIND="50"
REPORT_FILE=""
SOLVER=""

usage() {
    echo "Usage: $0 <file> [options]"
    echo ""
    echo "Comprehensive ESBMC security audit."
    echo ""
    echo "Options:"
    echo "  -t, --timeout <time>    Timeout per check (default: 120s)"
    echo "  -u, --max-unwind <n>    Maximum unwind bound (default: 50)"
    echo "  -r, --report <file>     Save report to file"
    echo "  -s, --solver <solver>   Use specific solver"
    echo "  -h, --help              Show this help"
    echo ""
    echo "The audit performs multiple passes:"
    echo "  1. Quick scan (low unwind, all default checks)"
    echo "  2. Memory safety (memory leaks, bounds, pointers)"
    echo "  3. Integer safety (overflow, underflow, division)"
    echo "  4. Concurrency (if applicable: deadlocks, races)"
    echo "  5. Deep verification (higher unwind bound)"
    echo "  6. K-induction (for proof attempts)"
    exit 1
}

# Parse arguments
FILE=""
EXTRA_OPTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -u|--max-unwind)
            MAX_UNWIND="$2"
            shift 2
            ;;
        -r|--report)
            REPORT_FILE="$2"
            shift 2
            ;;
        -s|--solver)
            SOLVER="--$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --contract)
            EXTRA_OPTS="$EXTRA_OPTS --contract $2"
            shift 2
            ;;
        -*)
            EXTRA_OPTS="$EXTRA_OPTS $1"
            shift
            ;;
        *)
            if [[ -z "$FILE" ]]; then
                FILE="$1"
            else
                EXTRA_OPTS="$EXTRA_OPTS $1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$FILE" ]]; then
    echo -e "${RED}Error: No input file specified${NC}"
    usage
fi

if [[ ! -f "$FILE" ]]; then
    echo -e "${RED}Error: File not found: $FILE${NC}"
    exit 1
fi

# Detect file type
EXT="${FILE##*.}"
LANG_OPTS=""
IS_CONCURRENT=0

case $EXT in
    c)
        # Check if file uses pthreads
        if grep -q "pthread" "$FILE" 2>/dev/null; then
            IS_CONCURRENT=1
        fi
        ;;
    cpp|cc|cxx)
        if grep -qE "pthread|std::thread|std::mutex" "$FILE" 2>/dev/null; then
            IS_CONCURRENT=1
        fi
        ;;
    py)
        if grep -qE "threading|multiprocessing" "$FILE" 2>/dev/null; then
            IS_CONCURRENT=1
        fi
        ;;
    sol)
        LANG_OPTS="--sol"
        ;;
    cu)
        LANG_OPTS="--32"
        IS_CONCURRENT=1
        ;;
esac

# Initialize report
REPORT=""
TOTAL_ISSUES=0
PASSED_CHECKS=0
FAILED_CHECKS=0

add_report() {
    REPORT="${REPORT}$1\n"
}

run_check() {
    local NAME="$1"
    local OPTS="$2"
    local UNWIND="$3"

    echo -e "${CYAN}[$NAME]${NC}"

    CMD="esbmc $LANG_OPTS $FILE --unwind $UNWIND --timeout $TIMEOUT $SOLVER $OPTS $EXTRA_OPTS"
    echo -e "  Command: ${BLUE}$CMD${NC}"

    START=$(date +%s)
    set +e
    OUTPUT=$($CMD 2>&1)
    set -e
    END=$(date +%s)
    ELAPSED=$((END - START))

    if echo "$OUTPUT" | grep -q "VERIFICATION SUCCESSFUL"; then
        echo -e "  Result: ${GREEN}PASSED${NC} (${ELAPSED}s)"
        add_report "  ✓ $NAME: PASSED (${ELAPSED}s)"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    elif echo "$OUTPUT" | grep -q "VERIFICATION FAILED"; then
        echo -e "  Result: ${RED}FAILED${NC} (${ELAPSED}s)"
        add_report "  ✗ $NAME: FAILED (${ELAPSED}s)"

        # Extract failure type
        if echo "$OUTPUT" | grep -q "array bounds"; then
            add_report "    - Array bounds violation"
        fi
        if echo "$OUTPUT" | grep -q "NULL pointer"; then
            add_report "    - Null pointer dereference"
        fi
        if echo "$OUTPUT" | grep -q "overflow"; then
            add_report "    - Integer overflow"
        fi
        if echo "$OUTPUT" | grep -q "memory leak"; then
            add_report "    - Memory leak"
        fi
        if echo "$OUTPUT" | grep -q "deadlock"; then
            add_report "    - Deadlock"
        fi
        if echo "$OUTPUT" | grep -q "data race"; then
            add_report "    - Data race"
        fi
        if echo "$OUTPUT" | grep -q "assertion"; then
            add_report "    - Assertion violation"
        fi

        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
        return 1
    else
        echo -e "  Result: ${YELLOW}UNKNOWN${NC} (${ELAPSED}s)"
        add_report "  ? $NAME: UNKNOWN (${ELAPSED}s)"
        return 2
    fi
}

# Print header
echo ""
echo "============================================"
echo -e "${CYAN}ESBMC Security Audit${NC}"
echo "============================================"
echo -e "File: ${BLUE}$FILE${NC}"
echo -e "Timeout per check: ${BLUE}$TIMEOUT${NC}"
echo -e "Max unwind: ${BLUE}$MAX_UNWIND${NC}"
if [[ $IS_CONCURRENT -eq 1 ]]; then
    echo -e "Concurrency: ${YELLOW}Detected${NC}"
fi
echo "============================================"
echo ""

add_report "ESBMC Security Audit Report"
add_report "============================"
add_report "File: $FILE"
add_report "Date: $(date)"
add_report ""
add_report "Results:"

# Pass 1: Quick Scan
echo -e "${YELLOW}Pass 1: Quick Scan${NC}"
add_report "\n[Pass 1: Quick Scan]"
run_check "Default checks" "" "5" || true
echo ""

# Pass 2: Memory Safety
echo -e "${YELLOW}Pass 2: Memory Safety${NC}"
add_report "\n[Pass 2: Memory Safety]"
run_check "Memory leak check" "--memory-leak-check" "10" || true
run_check "Bounds check" "" "15" || true
echo ""

# Pass 3: Integer Safety
echo -e "${YELLOW}Pass 3: Integer Safety${NC}"
add_report "\n[Pass 3: Integer Safety]"
run_check "Signed overflow" "--overflow-check" "10" || true
run_check "Unsigned overflow" "--unsigned-overflow-check" "10" || true
run_check "Shift UB" "--ub-shift-check" "10" || true
echo ""

# Pass 4: Concurrency (if applicable)
if [[ $IS_CONCURRENT -eq 1 ]]; then
    echo -e "${YELLOW}Pass 4: Concurrency Safety${NC}"
    add_report "\n[Pass 4: Concurrency Safety]"
    run_check "Deadlock check" "--deadlock-check --context-bound 2" "10" || true
    run_check "Data race check" "--data-races-check --context-bound 2" "10" || true
    echo ""
fi

# Pass 5: Deep Verification
echo -e "${YELLOW}Pass 5: Deep Verification${NC}"
add_report "\n[Pass 5: Deep Verification]"
run_check "Extended bounds" "--memory-leak-check --overflow-check" "$MAX_UNWIND" || true
echo ""

# Pass 6: K-Induction (attempt proof)
echo -e "${YELLOW}Pass 6: K-Induction Proof Attempt${NC}"
add_report "\n[Pass 6: K-Induction]"
run_check "K-induction" "--k-induction --max-k-step 20" "10" || true
echo ""

# Summary
echo "============================================"
echo -e "${CYAN}Audit Summary${NC}"
echo "============================================"
echo -e "Passed checks: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed checks: ${RED}$FAILED_CHECKS${NC}"
echo -e "Total issues found: ${RED}$TOTAL_ISSUES${NC}"
echo "============================================"

add_report "\n============================"
add_report "Summary"
add_report "============================"
add_report "Passed checks: $PASSED_CHECKS"
add_report "Failed checks: $FAILED_CHECKS"
add_report "Total issues: $TOTAL_ISSUES"

if [[ $TOTAL_ISSUES -eq 0 ]]; then
    echo -e "${GREEN}No issues detected in this audit.${NC}"
    add_report "\nNo issues detected."
else
    echo -e "${RED}$TOTAL_ISSUES issue(s) require attention.${NC}"
    add_report "\n$TOTAL_ISSUES issue(s) require attention."
fi

# Save report if requested
if [[ -n "$REPORT_FILE" ]]; then
    echo -e "$REPORT" > "$REPORT_FILE"
    echo -e "\nReport saved to: ${BLUE}$REPORT_FILE${NC}"
fi

# Exit with appropriate code
if [[ $TOTAL_ISSUES -gt 0 ]]; then
    exit 1
fi
exit 0
