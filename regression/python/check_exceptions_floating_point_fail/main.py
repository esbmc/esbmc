# FloatingPointError is a modelled type, so listing it is accepted, and the
# ZeroDivisionError listed beside it is still reported.
def main() -> None:
    b = nondet_int()
    q = 10 // b


main()
