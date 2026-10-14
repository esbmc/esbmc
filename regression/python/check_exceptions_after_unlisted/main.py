# The IndexError is only reachable after the ZeroDivisionError on the same
# path, which ends it, so the checked IndexError cannot escape.
def main() -> None:
    b = nondet_int()
    l = [1, 2]
    q = 10 // b
    if b == 0:
        x = l[5]


main()
