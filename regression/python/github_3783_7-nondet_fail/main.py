# popitem() returns the inserted value; asserting otherwise must fail
v: int = nondet_int()
__ESBMC_assume(v > 0)
d: dict[str, int] = {"a": v}
key, value = d.popitem()
assert value != v  # always false — VERIFICATION FAILED expected
