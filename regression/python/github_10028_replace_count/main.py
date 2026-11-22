n: int = nondet_int()
__ESBMC_assume(n >= -2 and n <= 3)
s = "xxx".replace("x", "y", n)
if n < 0:
    assert s == "yyy"
else:
    assert s.count("y") == min(n, 3)

k: int = nondet_int()
__ESBMC_assume(k == 5000000000)
assert "xx".replace("x", "y", k) == "yy"
