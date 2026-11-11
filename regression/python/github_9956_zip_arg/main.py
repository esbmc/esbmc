def mk(n: int):
    r = []
    for i in range(n):
        r.append(n - i)
    return r


def main() -> None:
    a = mk(2)
    s = list(zip(a, a))
    assert s[1][0] < s[0][0]


main()
