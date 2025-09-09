#!/bin/bash

cd regression/python

all_passed=true

# List of directories to ignore
ignored_dirs=(
  "AssertionError1_fail"
  "AssertionError2_fail"
  "branch_coverage-fail"
  "built-in-functions"
  "constants"
  "div6_fail"
  "div7_fail"
  "esbmc-assume"
  "enumerate15"
  "enumerate15_fail"
  "float1_fail"
  "func-no-params-types-fail"
  "function-option-fail"
  "global"
  "integer_squareroot_fail"
  "int_from_bytes"
  "input"
  "input2"
  "input3"
  "input4"
  "input5"
  "input4_fail"
  "input5_fail"
  "input6"
  "insertion_fail"
  "insertion3_fail"
  "jpl"
  "list9"
  "list10"
  "list15_fail"
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
  "random1"
  "random1-fail"
  "random2-fail"
  "range19-fail"
  "ternary_symbolic"
  "try-fail"
  "verifier-assume"
  "while-random-fail"
  "while-random-fail2"
  "while-random-fail3"
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
  (cd "$dir" && python3 main.py > /dev/null 2>&1)
  result=$?

  if [[ "$dir" == *fail* ]]; then
    if [ $result -eq 0 ]; then
      echo "❌ $dir: expected to fail, but executed successfully (exit 0)"
      all_passed=false
    else
      echo "✅ $dir: failed as expected (exit $result)"
    fi
  else
    if [ $result -eq 0 ]; then
      echo "✅ $dir: executed successfully (exit 0)"
    else
      echo "❌ $dir: expected to succeed, but failed (exit $result)"
      all_passed=false
    fi
  fi
done

if $all_passed; then
  echo -e "\n✅ All tests behaved as expected."
  exit 0
else
  echo -e "\n❌ Some tests did not behave as expected."
  exit 1
fi

cd -
