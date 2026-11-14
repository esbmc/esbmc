class uint64(int):
    pass


def half(n: uint64) -> uint64:
    x = n
    assert x >= 1
    return (x + 1) // 2


assert half(uint64(2**63 + 5)) == 2**62 + 3
