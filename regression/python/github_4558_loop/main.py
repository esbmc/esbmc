from stubs import *
from kernel import f

t4: Tile4D = Tile4D(5, 4, 50, 100)
src: Tile = Tile(50, 100)
f(t4, src)
