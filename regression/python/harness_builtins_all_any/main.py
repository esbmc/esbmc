# Verification harness for all/any (src/python-frontend/models/builtins.py).
#
# all(l) is True iff every element is truthy (True for the empty list);
# any(l) is True iff some element is truthy (False for the empty list).
#
# REQUIRES:
#   R1: three fully non-deterministic booleans, covering all 8 combinations.
#
# ENSURES (av = all(l), ov = any(l)):
#   E1: av == (a and b and c)                   [conjunction semantics]
#   E2: ov == (a or b or c)                     [disjunction semantics]
#   E3: av implies ov                           [non-empty list: all => any]
#
# The all()/any() results are cached in locals so each model loop is
# symex-unwound once, keeping the harness fast.
a: bool = nondet_bool()
b: bool = nondet_bool()
c: bool = nondet_bool()
l: list[bool] = [a, b, c]

av: bool = all(l)
ov: bool = any(l)

assert av == (a and b and c)  # E1
assert ov == (a or b or c)  # E2

if av:
    assert ov  # E3
