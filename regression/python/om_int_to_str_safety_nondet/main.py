def nondet_int() -> int: ...

# __python_int_to_str: 21-byte alloca buffer; nondet v in [0..999].
# Verifies no OOB/overflow and that the decimal length is in [1,3].
v = nondet_int()
__ESBMC_assume(v >= 0 and v <= 999)
s = str(v)
assert len(s) >= 1
assert len(s) <= 3
assert s[0] >= '0' and s[0] <= '9'
