from stubs import *


def f(t3: Tile3D, src: Tile, bm: int, k: int) -> None:
    copy(t3[bm * k, :, :], src)
