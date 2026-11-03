# An explicit raise AssertionError ends the program just as every other
# exception does, so the out-of-bounds read below it is unreachable.
def main():
    l = [0, 1]
    i = nondet_int()
    if not (0 <= i and i < 2):
        raise AssertionError
    x = l[i]
    assert x >= 0


main()
