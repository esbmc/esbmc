#!/bin/sh
# A local model solver that answers (check-sat) with sat and then exits,
# dying before the (get-value) queries used to read out the counterexample.
# It never acknowledges the :print-success protocol commands either, so the
# backend drops it during negotiation: no counterexample can be built, and
# the run must fail cleanly rather than hang or trust a model it cannot read.
while IFS= read -r line; do
  case "$line" in
    *'(check-sat)'*) echo sat; exit 0 ;;
  esac
done
