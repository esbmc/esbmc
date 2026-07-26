#!/bin/sh
# Stand-in for a NeuroSym run that prints only log noise and never emits a
# sat/unsat/unknown verdict; ESBMC must not invent a result.
echo "[NeuroSym] loading model..."
echo "[NeuroSym] no verdict on any line"
