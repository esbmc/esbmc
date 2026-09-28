#!/bin/sh
# Stand-in for an SMT-LIB solver that cannot decide anything: it answers every
# (check-sat) with an error.
while IFS= read -r line; do
  [ "$line" = "(check-sat)" ] && echo '(error "fake solver failure")'
done
exit 0
