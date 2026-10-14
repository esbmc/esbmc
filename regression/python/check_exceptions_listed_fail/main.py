# The same program with ZeroDivisionError checked fails at the division.
def main() -> None:
    b = nondet_int()
    q = 10 // b
    l = [1, 2]
    x = l[b % 2]


main()
