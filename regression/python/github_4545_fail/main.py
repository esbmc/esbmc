from stubs import *

def f(t: Tile) -> Tile:
    return t[0:128, 0:512]

t: Tile = Tile(256, 512)
r: Tile = f(t)
assert r.d0 == 999
