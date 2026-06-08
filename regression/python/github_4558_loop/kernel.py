from stubs import *


def f(t4: Tile4D, src: Tile) -> None:
    for i in range(2):
        copy(t4[i, 0, :, :], src)
