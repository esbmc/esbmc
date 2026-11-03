# assert ok(i) raises AssertionError when i is out of range, so l[i] is
# only reached in range.
def ok(i: int) -> bool:
    return 0 <= i and i < 2


def main():
    l = [0, 1]
    i = nondet_int()
    assert ok(i)
    x = l[i]


main()
