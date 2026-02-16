x = nondet_str()
__ESBMC_assume(len(x) < 8)

y = nondet_int()
__ESBMC_assume(y>=0 and y <= 10)

z = nondet_float()
__ESBMC_assume(z >= 0 and z <= 10)

l = [x, y, z]

a = l.pop()
assert a == z

b = l.pop()
assert b == y 

c = l.pop()
assert c == x
