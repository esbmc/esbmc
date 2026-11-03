def f() -> int:
    return 1


def g() -> int:
    return 2


def main() -> None:
    ls = [f]
    ls[0] = g
    assert ls[0]() == 2
    ls.insert(0, f)
    assert ls[0]() == 1
    ls.append(g)
    assert ls[2]() == 2


main()
