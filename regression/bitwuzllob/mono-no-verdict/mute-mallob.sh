#!/bin/sh
# Stand-in for a Mallob run that prints only log noise and never emits a
# sat/unsat/unknown verdict; ESBMC must not invent a result.
echo "c fake-mallob produced only log noise"
echo "c no verdict on any line"
