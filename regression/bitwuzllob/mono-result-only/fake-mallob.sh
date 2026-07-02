#!/bin/sh
# Stand-in for `mallob -mono=<file> -mono-app=SMT`: one-shot solve of an
# SMT-LIB2 file, verdict on stdout amid log noise (bare SMT-LIB style).
f=""
for arg in "$@"; do case "$arg" in -mono=*) f="${arg#-mono=}";; esac; done
[ -n "$f" ] || f="$1"
echo "c fake-mallob solving $f"
z3 -smt2 "$f" | tail -n 1
