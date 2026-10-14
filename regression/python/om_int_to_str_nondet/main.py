def nondet_int() -> int: ...

# Exercises negative values and zero — the sign-handling and "0" special-case
# paths in __python_int_to_str, constrained to [-99..99].
v = nondet_int()
__ESBMC_assume(v >= -99 and v <= 99)
s = str(v)
# Negative: leading '-', length 2..3
# Zero: "0", length 1
# Positive: length 1..2
if v < 0:
    assert s[0] == '-'
    assert len(s) >= 2
    assert len(s) <= 3
elif v == 0:
    assert s == "0"
else:
    assert len(s) >= 1
    assert len(s) <= 2
    assert s[0] >= '1' and s[0] <= '9'
