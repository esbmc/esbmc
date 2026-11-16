def mk(n: int):
    r = []
    for i in range(n):
        r.append(n - i)
    return r


def main() -> None:
    s = sorted(mk(3))
    assert s[0] > s[1]


main()
