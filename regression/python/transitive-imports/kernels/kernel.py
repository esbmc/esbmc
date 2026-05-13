from stubs import Tile, make_tile, check_shape


def add_kernel(a: Tile, b: Tile) -> Tile:
    check_shape(a, b)
    return make_tile(a.d0, a.d1)
