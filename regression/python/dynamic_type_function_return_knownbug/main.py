# A function's return type diverges across branches (int vs str) based on a
# nondet condition. Unlike a local variable, no retyping heuristic runs on
# RETURN at all: the GOTO for f emits `RETURN: 1` on one path and
# `RETURN: { 97, 0 }` on the other, with no attempt to reconcile the two C
# types. This crashes goto-symex/SMT encoding
# ("ERROR: Unexpected type in int/ptr typecast") rather than producing a
# wrong verdict. The assertion holds on both branches, so the correct verdict,
# once fixed, is VERIFICATION SUCCESSFUL.

def f(cond):
    if cond:
        return 1
    else:
        return "a"

x = f(nondet_bool())
assert x == 1 or x == "a"
