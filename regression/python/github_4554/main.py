from stubs import *
from tensor_add_kernel import nki_tensor_add

t: Tile = Tile(256, 256)
r: Tile = nki_tensor_add(t)
assert r.d0 == 128
