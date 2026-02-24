#!/bin/bash

cd regression/python

# Activate virtual environment if it exists (from build-with-venv.sh)
# CI builds will have dependencies installed by build.sh, so this is optional
if [ -f "../esbmc-venv/bin/activate" ]; then
    source ../esbmc-venv/bin/activate
fi

all_passed=true
failed_tests=()

# List of directories to ignore
ignored_dirs=(
  "AssertionError1_fail"
  "AssertionError2_fail"
  "branch_coverage-fail"
  "built-in-functions"
  "cover1"
  "cover2"
  "cover3"
  "cover4"
  "cover5"
  "convert-byte-update2"
  "constants"
  "decimal"
  "decimal_fail"
  "decimal4"
  "decimal4_fail"
  "dict_del12_fail"
  "dict_del13_fail"
  "dict_del14"
  "dict_del14_fail"
  "div6_fail"
  "div7_fail"
  "esbmc-assume"
  "enumerate15"
  "enumerate15_fail"
  "func-no-params-types-fail"
  "function-option-fail"
  "github_2843_fail"
  "github_2843_4_fail"
  "github_2908_1"
  "github_2908_2"
  "github_2993_fail"
  "github_2993_2_fail"
  "github_3012_3_fail"
  "github_3090_4"
  "github_3090_4_fail"
  "github_3090_5"
  "github_3090_5_fail"
  "github_3313_3"
  "github_3337_2"
  "github_3337_3"
  "github_3337_4"
  "github_3560"
  "github_3560_1"
  "github_3560_3"
  "github_3560_4"
  "global"
  "infer-func-no-return_fail"
  "integer_squareroot_fail"
  "int_from_bytes"
  "input"
  "input2"
  "input3"
  "input5"
  "input6"
  "insertion_fail"
  "insertion3_fail"
  "jpl"
  "jpl_1"
  "list9"
  "list10"
  "list15_fail"
  "list-repetition-symbolic"
  "loop-invariant"
  "loop-invariant2"
  "min_max_3_fail"
  "missing-return_fail"
  "missing-return7_fail"
  "missing-return8_fail"
  "missing-return9_fail"
  "missing-return10_fail"
  "missing-return11_fail"
  "missing-return12_fail"
  "missing-return13_fail"
  "missing-return14_fail"
  "neural-net_fail"
  "problem1_fail"
  "random1"
  "random1-fail"
  "random2-fail"
  "random6_fail"
  "range19-fail"
  "ternary_symbolic"
  "try-fail"
  "verifier-assume"
  "while-random-fail"
  "while-random-fail2"
  "while-random-fail3"
  "github_3181_fail"
  "incremental-smt-loop-off"
  "incremental-smt-loop-on"
  "incremental-smt-assert-pass"
  "type-annotation-check"
  "type-annotation-generics-fail"
  "string-char-symbolic-success"
  "string-symbolic-1"
  "string-symbolic-2"
  "string-symbolic-3"
  "string-symbolic-4"
  "string-symbolic-7"
  "string-symbolic-8"
)

for dir in */; do
  # Remove trailing slash
  dir="${dir%/}"

  # Skip if main.py does not exist
  if [ ! -f "$dir/main.py" ]; then
    continue
  fi

  # Skip if directory name contains "nondet"
  if echo "$dir" | grep -iq 'nondet'; then
    echo "🚫 IGNORED: $dir (contains 'nondet')"
    continue
  fi

  # Skip if in the ignore list
  for ignored in "${ignored_dirs[@]}"; do
    if [[ "$dir" == "$ignored" ]]; then
      echo "🚫 IGNORED: $dir (in ignore list)"
      continue 2  # Skip this iteration of the outer loop
    fi
  done

  echo ">>> Testing $dir"

  # Run the script and capture the exit code
  # Use virtual environment's python if activated
  # Otherwise, on macOS use Python 3.12 (matching build.sh which installs python@3.12)
  if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="python"
  elif [[ "$OSTYPE" == "darwin"* ]] && command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
  else
    PYTHON_CMD="python3"
  fi
  (cd "$dir" && $PYTHON_CMD main.py > /dev/null 2>&1)
  result=$?

  if [[ "$dir" == *fail* ]]; then
    if [ $result -eq 0 ]; then
      echo "❌ $dir: expected to fail, but executed successfully (exit 0)"
      all_passed=false
      failed_tests+=("$dir")
    else
      echo "✅ $dir: failed as expected (exit $result)"
    fi
  else
    if [ $result -eq 0 ]; then
      echo "✅ $dir: executed successfully (exit 0)"
    else
      echo "❌ $dir: expected to succeed, but failed (exit $result)"
      all_passed=false
      failed_tests+=("$dir")
    fi
  fi
done

if [ ${#failed_tests[@]} -eq 0 ]; then
  echo -e "\n✅ All tests behaved as expected."
  exit 0
else
  echo -e "\n❌ Some tests did not behave as expected."
  echo "Failed tests:"
  for test in "${failed_tests[@]}"; do
    echo " - $test"
  done
  exit 1
fi

cd -
