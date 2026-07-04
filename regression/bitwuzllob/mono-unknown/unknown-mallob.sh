#!/bin/sh
# Stand-in for a Mallob run that finishes without deciding the formula: it
# emits the SAT-competition-style "s UNKNOWN" verdict amid log noise.
echo "c fake-mallob solving $1"
echo "s UNKNOWN"
