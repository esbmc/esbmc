x:str = nondet_str()
__ESBMC_assume(len(x) < 8)

y:int = nondet_int()
__ESBMC_assume(y>=0 and y <= 10)

z:float = nondet_float()
__ESBMC_assume(z >= 0 and z <= 10)

l = [x, y, z]

a:float = l.pop()
assert a == z

b:int = l.pop()
assert b == y 

c:str = l.pop()
assert c == x
