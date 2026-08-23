def tail(n):
    if n > 0:
        r = tail(n - 1)
        return r
    return 0


def fact(n):
    if n > 1:
        r = fact(n - 1)
        return r * n
    return 1


def build(n):
    if n > 0:
        r = build(n - 1)
        return r
    return [[]]


def main():
    assert tail(3) == 0
    assert fact(5) == 120
    assert len(build(2)) == 1


main()
