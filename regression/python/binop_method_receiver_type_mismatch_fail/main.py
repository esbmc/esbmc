class uint64(int):
    pass


def f(index: uint64) -> int:
    position: uint64 = index + 1
    return (position // 256).bit_length()


assert f(uint64(300)) == 2
