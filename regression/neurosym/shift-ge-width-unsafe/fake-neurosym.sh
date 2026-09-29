#!/bin/sh
# Stand-in for NeuroSym's batch mode, emitting the (model (define-fun ...))
# block its format_output() prints on a sat verdict. The model block is what
# drives the backend's local evaluator; without it every (get-value) would go
# to --neurosym-model-prog and this test would not exercise the evaluator.
q=$(mktemp)   # no template: macOS mktemp needs the X run at the end
cat "$1" > "$q"
echo '(get-model)' >> "$q"
out=$(z3 -smt2 "$q" 2>/dev/null)
rm -f "$q"
# The one-shot backend sets :print-success, so z3 acknowledges every
# set-option with "success" before the verdict. Drop those first: the old
# --neurosym backend sent no such options and this read line 1 directly.
out=$(printf '%s\n' "$out" | grep -vx success)
verdict=$(printf '%s\n' "$out" | head -n 1)
echo "[NeuroSym] solving $1"
echo "$verdict"
if [ "$verdict" = sat ]; then
  printf '%s\n' "$out" | tail -n +2 | sed '1s/^(/(model /'
fi
