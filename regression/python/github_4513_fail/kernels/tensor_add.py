from stubs import Tile, make_tile

def nki_tensor_add(a: Tile, b: Tile) -> Tile:
    assert a.d0 == b.d0
    return make_tile(a.d0, a.d1)
