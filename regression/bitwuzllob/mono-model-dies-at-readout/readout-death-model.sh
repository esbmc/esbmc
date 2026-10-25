#!/bin/sh
# A local model solver that behaves correctly right up to the counterexample:
# it acknowledges the :print-success protocol commands, answers (check-sat)
# with sat, and only then dies -- standing in for a solver that crashes or is
# OOM-killed while the (get-value) queries are being served. The verdict is
# already settled at that point, so the backend must report the dead process
# rather than substitute default values into the trace.
while IFS= read -r line; do
  case "$line" in
    *'(check-sat)'*) echo sat ;;
    *'(get-value'*)  exit 1 ;;
    *'(exit)'*)      exit 0 ;;
    '')              ;;
    *)               echo success ;;
  esac
done
