# #10428
x: float = nondet_float()
y: float = nondet_float()
__ESBMC_assume(x == x and y == y)
a = [x, y]
c = a + []
assert c[1] == x
