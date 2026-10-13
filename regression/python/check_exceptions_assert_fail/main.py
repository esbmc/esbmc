# Checking Exception covers AssertionError, so the assert stays a property.
def main() -> None:
    l = [0, 1]
    i = nondet_int()
    assert 0 <= i and i < 2
    x = l[i]


main()
