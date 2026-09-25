#!/bin/sh
# Stand-in for `python main.py <file>`: one-shot solve of an SMT-LIB2 file,
# verdict on stdout amid log noise, as NeuroSym's batch mode produces.
echo "[NeuroSym] loading model..."
echo "[NeuroSym] solving $1"
z3 -smt2 "$1" | tail -n 1
