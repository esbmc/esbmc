from stubs import make_tile
from kernels.kernel import add_kernel

a = make_tile(4, 8)
b = make_tile(4, 8)
c = add_kernel(a, b)
assert c.d0 == 4
assert c.d1 == 8
