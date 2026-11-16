def mk(n: int):
    r = []
    for i in range(n):
        r.append(n - i)
    return r


def main() -> None:
    s = reversed(mk(3))
    t = list(s)
    assert t[0] < t[1]


main()
