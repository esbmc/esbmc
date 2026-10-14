from stubs import make_tile
from kernels.tensor_add import nki_tensor_add

a = make_tile(4, 8)
c = nki_tensor_add(a, a)
assert c.d0 == 4
