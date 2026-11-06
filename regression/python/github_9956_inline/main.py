def mk(n: int):
    r = []
    for i in range(n):
        r.append(n - i)
    return r


def main() -> None:
    assert sorted(mk(3))[0] <= sorted(mk(3))[1]


main()
