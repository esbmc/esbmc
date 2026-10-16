# list.pop() on an empty list, or with an index out of range, raises an
# IndexError that except IndexError catches (#8504).
def main() -> None:
    s: list[int] = []
    if nondet_bool():
        s.append(1)
    try:
        x = s.pop()
    except IndexError:
        x = 0
    assert x == 0 or x == 1

    t = [4, 5]
    k = nondet_int()
    __ESBMC_assume(-4 <= k and k <= 3)
    try:
        y = t.pop(k)
    except IndexError:
        y = 0
    assert y == 0 or y == 4 or y == 5


main()
