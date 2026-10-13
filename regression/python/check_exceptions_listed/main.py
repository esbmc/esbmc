# Only IndexError is checked. The ZeroDivisionError at b == 0 ends its path,
# and b % 2 is 0 or 1, so no IndexError can escape.
def main() -> None:
    b = nondet_int()
    q = 10 // b
    l = [1, 2]
    x = l[b % 2]


main()
