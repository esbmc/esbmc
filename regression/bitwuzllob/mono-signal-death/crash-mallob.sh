#!/bin/sh
# Stand-in for a Mallob run that prints a verdict and then dies abnormally
# (crash, OOM kill, mpirun rank failure). ESBMC must not trust the verdict.
echo "s UNSATISFIABLE"
kill -SEGV $$
