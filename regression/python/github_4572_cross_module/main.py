from stubs import Tile3D


def outer(slot: int) -> None:
    buf: Tile3D = Tile3D(2)
    src: Tile3D = Tile3D(1)

    def closure(s: int) -> None:
        buf[s] = src

    closure(slot)


outer(0)
