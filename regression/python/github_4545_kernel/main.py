from stubs import *

def kernel(a_input: Tile) -> Tile:
    return a_input[0:128, 0:512]

def driver() -> None:
    a: Tile = Tile(256, 512)
    r: Tile = kernel(a)
    assert r.d0 == 128

driver()
