# The unchecked ValueError ends the paths where i is out of range, and the
# forward condition must see that to prove l[i] in bounds.
def main() -> None:
    l = [0, 1]
    i = nondet_int()
    if i < 0 or i > 1:
        raise ValueError
    x = l[i]


main()
