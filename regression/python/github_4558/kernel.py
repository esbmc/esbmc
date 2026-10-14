from stubs import *


def f(t3: Tile3D, src: Tile, i: int) -> None:
    copy(t3[i, :, :], src)
