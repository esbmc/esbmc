#!/bin/sh
# Stand-in for a Mallob run that reports unsat regardless of the formula, so
# the (unsat) verdict comes from Mallob alone and the local model solver — here
# a dead one — is never needed. z3-free and deterministic.
echo "c fake-mallob always claims unsat"
echo "s UNSATISFIABLE"
