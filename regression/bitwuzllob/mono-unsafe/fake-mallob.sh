#!/bin/sh
# Stand-in for `mallob -mono=<file> -mono-app=SMT`, emitting the
# SAT-competition-style "s ..." verdict lines Mallob uses.
f=""
for arg in "$@"; do case "$arg" in -mono=*) f="${arg#-mono=}";; esac; done
[ -n "$f" ] || f="$1"
echo "c fake-mallob solving $f"
case "$(z3 -smt2 "$f" | tail -n 1)" in
  sat) echo "s SATISFIABLE";;
  unsat) echo "s UNSATISFIABLE";;
  *) echo "s UNKNOWN";;
esac
