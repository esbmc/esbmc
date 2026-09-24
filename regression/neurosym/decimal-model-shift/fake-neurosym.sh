#!/bin/sh
# Stand-in emitting its model as plain decimal rather than #x hex -- a form
# NeuroSym's format_output() and numeric_value() both accept. A bit pattern
# with the top bit set then arrives as a negative numeral, which must not
# escape local_eval_bv()'s width masking: unmasked it slips past the shift
# guards into power(2, negative), which asserts (or loops, with NDEBUG).
q=$(mktemp)
cat "$1" > "$q"; echo '(get-model)' >> "$q"
out=$(z3 -smt2 "$q" 2>/dev/null); rm -f "$q"
# The one-shot backend sets :print-success, so z3 acknowledges every
# set-option with "success" before the verdict. Drop those first: the old
# --neurosym backend sent no such options and this read line 1 directly.
out=$(printf '%s\n' "$out" | grep -vx success)
verdict=$(printf '%s\n' "$out" | head -n 1)
echo "[NeuroSym] solving $1"
echo "$verdict"
[ "$verdict" = sat ] || exit 0
printf '%s\n' "$out" | tail -n +2 | sed '1s/^(/(model /' | python3 -c '
import sys, re
def signed(m):
    h = m.group(1); w = len(h) * 4; u = int(h, 16)
    return str(u - (1 << w)) if u >> (w - 1) else str(u)
sys.stdout.write(re.sub(r"#x([0-9a-fA-F]+)", signed, sys.stdin.read()))'
