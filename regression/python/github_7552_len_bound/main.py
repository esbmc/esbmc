# len() of a name bound to chr() counts code points, not the UTF-8 bytes
# chr() stores (#7552).
c = chr(200)
assert len(c) == 1
e = chr(0x1F600)
assert len(e) == 1
n = nondet_int()
__ESBMC_assume(0 <= n and n <= 1114111)
r = chr(n)
assert len(r) == 1
