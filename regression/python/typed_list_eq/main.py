def countdown(n: int) -> list[int]:
    out = []
    i = n
    while i > 0:
        i -= 1
        out.append(i)
    return out


def main() -> None:
    n = nondet_int()
    if n < 0 or n > 3:
        return
    a = [0] * n
    b = []
    i = 0
    while i < n:
        b.append(0)
        i += 1
    assert a == b
    assert not (a != b)
    b.append(1)
    assert a != b
    a.append(2)
    assert a != b
    c = countdown(n)
    assert countdown(n) == c
    assert countdown(n) == sorted(c, reverse=True)
    assert sorted(c) != countdown(n) or n < 2


main()
