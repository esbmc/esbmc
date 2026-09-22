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
verdict=$(printf '%s\n' "$out" | head -n 1)
echo "[NeuroSym] solving $1"
echo "$verdict"
if [ "$verdict" = sat ]; then
  printf '%s\n' "$out" | tail -n +2 | sed '1s/^(/(model /'
fi
