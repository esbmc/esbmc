def mk(n: int):
    r = []
    for i in range(n):
        r.append(n - i)
    return r


def main() -> None:
    a = b = sorted(mk(3))
    assert a[0] < b[1]


main()
