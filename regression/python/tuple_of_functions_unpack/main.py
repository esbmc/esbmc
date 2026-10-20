def f1() -> int:
    return 1


def f2() -> int:
    return 2


def main() -> None:
    fs = (f1, f2)
    a, b = fs
    assert a() + b() == 3
    assert fs[1]() == 2
    x = (f1, 3)
    g, n = x
    assert g() + n == 4
    a, b = c, d = f1, f2
    assert c() == 1 and d() == 2 and b() == 2


main()
