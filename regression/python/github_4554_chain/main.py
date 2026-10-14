from stubs import Tile
from outer_kernel import outer

t: Tile = Tile(128, 128)
r: Tile = outer(t)
assert r.d0 == 64
