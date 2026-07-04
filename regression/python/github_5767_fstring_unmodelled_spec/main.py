# Tier 2 (nondet generalisation) for #5767: the fallback for an unmodelled
# f-string format spec ('e' here) must be usable as an operand of == without
# aborting GOTO conversion, in any expression position -- not just inside an
# `assert`. `b or not b` is a CPython-true tautology regardless of what `b`
# evaluates to, so this is sound for every possible fallback value; it also
# forces the `==` comparison to actually lower into the GOTO program (see
# --goto-functions-only) rather than being optimised away before conversion.
w = -1.5
b = (f"{w:e}" == "x")
assert b or not b
