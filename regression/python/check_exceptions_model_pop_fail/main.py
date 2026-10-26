# The same program with IndexError checked fails at the second pop().
def main() -> None:
    l = [1]
    if nondet_bool():
        l.pop()
    n = len(l)
    x = l.pop()
    if n == 0:
        y = 10 // 0


main()
