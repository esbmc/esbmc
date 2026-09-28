#!/bin/sh
# Stand-in for an SMT-LIB solver that refutes every formula it is given.
while IFS= read -r line; do
  [ "$line" = "(check-sat)" ] && echo unsat
done
exit 0
