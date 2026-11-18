import re

n: int = nondet_int()
__ESBMC_assume(n >= 0 and n <= 3)
assert re.sub("a", "b", "aaaa", n) == "bbbb"
