# #10428
x: float = nondet_float()
y: float = nondet_float()
__ESBMC_assume(x == x and y == y)
a = [x, y]
b = a[:]
c = a + []
assert b[0] == x and b[1] == y
assert c[0] == x and c[1] == y
