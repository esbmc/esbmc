#!/bin/sh
# A local model solver that answers (check-sat) with sat and then exits,
# dying before the (get-value) queries used to read out the counterexample.
while IFS= read -r line; do
  case "$line" in
    *'(check-sat)'*) echo sat; exit 0 ;;
  esac
done
