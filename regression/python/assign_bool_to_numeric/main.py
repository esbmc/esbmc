# A bool rebinding an int or float variable is converted to its value.
def main() -> None:
    x = 1.5
    x = True
    assert x == 1
    assert x + 1 == 2
    n = 7
    n = False
    assert n == 0


main()
