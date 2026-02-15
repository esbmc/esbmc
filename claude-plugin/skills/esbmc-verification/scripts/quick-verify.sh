#!/bin/bash
#
# quick-verify.sh - Quick ESBMC verification wrapper
#
# Usage: ./quick-verify.sh <file> [options]
#
# Provides sensible defaults for quick bug hunting with ESBMC.
# Supports C, C++, Python, and Solidity files.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
TIMEOUT="60s"
UNWIND="10"
SOLVER=""  # Use default

usage() {
    echo "Usage: $0 <file> [options]"
    echo ""
    echo "Quick ESBMC verification with sensible defaults."
    echo ""
    echo "Options:"
    echo "  -t, --timeout <time>    Set timeout (default: 60s)"
    echo "  -u, --unwind <n>        Set loop unwind bound (default: 10)"
    echo "  -s, --solver <solver>   Set solver (z3, bitwuzla, boolector)"
    echo "  -m, --memory            Enable memory leak checking"
    echo "  -o, --overflow          Enable overflow checking"
    echo "  -c, --concurrent        Enable concurrency checks"
    echo "  -a, --all               Enable all safety checks"
    echo "  -v, --verbose           Show ESBMC output"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 program.c                  # Quick verification"
    echo "  $0 program.c -m -o            # With memory and overflow checks"
    echo "  $0 program.c -a -t 5m         # All checks, 5 minute timeout"
    echo "  $0 contract.sol --contract MyContract"
    exit 1
}

# Parse arguments
FILE=""
EXTRA_OPTS=""
MEMORY_CHECK=""
OVERFLOW_CHECK=""
CONCURRENT_CHECK=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -u|--unwind)
            UNWIND="$2"
            shift 2
            ;;
        -s|--solver)
            SOLVER="--$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY_CHECK="--memory-leak-check"
            shift
            ;;
        -o|--overflow)
            OVERFLOW_CHECK="--overflow-check --unsigned-overflow-check"
            shift
            ;;
        -c|--concurrent)
            CONCURRENT_CHECK="--deadlock-check --data-races-check --context-bound 2"
            shift
            ;;
        -a|--all)
            MEMORY_CHECK="--memory-leak-check"
            OVERFLOW_CHECK="--overflow-check --unsigned-overflow-check"
            CONCURRENT_CHECK="--deadlock-check --data-races-check --context-bound 2"
            shift
            ;;
        -v|--verbose)
            VERBOSE="1"
            shift
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

# Check if file provided
if [[ -z "$FILE" ]]; then
    echo -e "${RED}Error: No input file specified${NC}" >&2
    usage
fi

# Check if file exists
if [[ ! -f "$FILE" ]]; then
    echo -e "${RED}Error: File not found: $FILE${NC}"
    exit 1
fi

# Detect file type and set appropriate options
EXT="${FILE##*.}"
LANG_OPTS=""

case $EXT in
    c)
        echo -e "${BLUE}Verifying C file: $FILE${NC}"
        ;;
    cpp|cc|cxx)
        echo -e "${BLUE}Verifying C++ file: $FILE${NC}"
        ;;
    py)
        echo -e "${BLUE}Verifying Python file: $FILE${NC}"
        ;;
    sol)
        echo -e "${BLUE}Verifying Solidity file: $FILE${NC}"
        LANG_OPTS="--sol"
        ;;
    cu)
        echo -e "${BLUE}Verifying CUDA file: $FILE${NC}"
        LANG_OPTS="--32"
        ;;
    *)
        echo -e "${YELLOW}Warning: Unknown file extension, assuming C${NC}"
        ;;
esac

# Build command
CMD="esbmc"
[[ -n "$LANG_OPTS" ]] && CMD="$CMD $LANG_OPTS"
CMD="$CMD $FILE"
CMD="$CMD --unwind $UNWIND"
CMD="$CMD --timeout $TIMEOUT"
[[ -n "$SOLVER" ]] && CMD="$CMD $SOLVER"
[[ -n "$MEMORY_CHECK" ]] && CMD="$CMD $MEMORY_CHECK"
[[ -n "$OVERFLOW_CHECK" ]] && CMD="$CMD $OVERFLOW_CHECK"
[[ -n "$CONCURRENT_CHECK" ]] && CMD="$CMD $CONCURRENT_CHECK"
[[ -n "$EXTRA_OPTS" ]] && CMD="$CMD $EXTRA_OPTS"

echo -e "${BLUE}Running: $CMD${NC}"
echo ""

# Run ESBMC
START_TIME=$(date +%s)

set +e
OUTPUT=$($CMD 2>&1)
RESULT=$?
set -e

if [[ -n "$VERBOSE" ]]; then
    echo "$OUTPUT"
else
    echo "$OUTPUT" | tail -20
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "----------------------------------------"

if [[ $RESULT -eq 0 ]]; then
    echo -e "${GREEN}✓ VERIFICATION SUCCESSFUL${NC} (${ELAPSED}s)"
else
    echo -e "${RED}✗ VERIFICATION FAILED${NC} (${ELAPSED}s)"

    # Provide hints based on output
    if echo "$OUTPUT" | grep -q "array bounds"; then
        echo -e "${YELLOW}Hint: Array bounds violation detected. Check array indices.${NC}"
    fi
    if echo "$OUTPUT" | grep -q "pointer"; then
        echo -e "${YELLOW}Hint: Pointer error detected. Check for null pointers or invalid access.${NC}"
    fi
    if echo "$OUTPUT" | grep -q "overflow"; then
        echo -e "${YELLOW}Hint: Integer overflow detected. Add bounds checking.${NC}"
    fi
    if echo "$OUTPUT" | grep -q "memory leak"; then
        echo -e "${YELLOW}Hint: Memory leak detected. Ensure all allocations are freed.${NC}"
    fi
    if echo "$OUTPUT" | grep -q "deadlock"; then
        echo -e "${YELLOW}Hint: Potential deadlock. Check lock ordering.${NC}"
    fi
    if echo "$OUTPUT" | grep -q "data race"; then
        echo -e "${YELLOW}Hint: Data race detected. Add synchronization.${NC}"
    fi
fi

exit $RESULT
