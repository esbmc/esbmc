#!/bin/sh
# Stand-in for a NeuroSym run that (wrongly) claims sat regardless of the
# formula, so the local model solver, replaying an actually-unsat formula,
# disagrees. ESBMC must refuse to build a counterexample from divergence.
echo "[NeuroSym] always claims sat"
echo "sat"
