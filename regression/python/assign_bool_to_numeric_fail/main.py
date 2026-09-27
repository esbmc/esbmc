# A bool rebinding an int or float variable is converted to its value.
def main() -> None:
    x = 1.5
    x = True
    n = 7
    n = False
    assert x == 0 or n == 7


main()
