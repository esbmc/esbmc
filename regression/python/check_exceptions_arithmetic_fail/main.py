# Listing ArithmeticError checks its subclass ZeroDivisionError too.
def main() -> None:
    b = nondet_int()
    q = 10 // b


main()
