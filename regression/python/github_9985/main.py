# Issue #9985: zip() and enumerate() of a call, and sorted(map(...)).
def mk(n: int):
    r = []
    for i in range(n):
        r.append(n - i)
    return r


def main() -> None:
    s = list(zip(mk(2), mk(3)))
    assert len(s) == 2
    assert s[1][0] < s[0][0]
    assert s[0][1] == 3
    e = list(enumerate(mk(2), 1))
    assert e[1] == (2, 1)
    a = [-3, 1, -2]
    t = sorted(map(abs, a))
    assert t == [1, 2, 3]


main()
