def f():
    return 1


def g(x: int) -> int:
    return x * 2


def main():
    ls = [f, g]
    ls2 = ls[0:1]
    c = ls2[0]()
    assert c == 1
    tail = ls[1:]
    assert tail[0](3) == 6


main()
