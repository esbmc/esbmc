def f(x):
    return x + 1


def main() -> None:
    n = nondet_int()
    r = map(f, [1, n])
    assert list(r) == [2, n + 1]

    a, b = map(lambda v: v * 2, [n, 3])
    assert a == 2 * n and b == 6

    total = 0
    for v in map(f, [1, 2]):
        total += v
    else:
        total += 10
    assert total == 15


main()
