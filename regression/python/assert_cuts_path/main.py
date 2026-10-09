# A failing assert raises AssertionError and ends the program, so a later
# exception on the same path is unreachable: l[i] is only out of bounds when
# the assert's condition is false.
def main():
    l = [0, 1]
    i = nondet_int()
    assert 0 <= i and i < 2
    x = l[i]
    assert x >= 0


main()
