def f(c: bool) -> int:
    if c:
        k = 1
    return k


def last(n: int) -> int:
    for i in range(n):
        v = i
    return v


def main() -> None:
    assert f(True) == 1
    assert last(3) == 2
    try:
        f(False)
        assert False
    except UnboundLocalError:
        pass
    try:
        last(0)
        assert False
    except NameError:
        pass


main()
