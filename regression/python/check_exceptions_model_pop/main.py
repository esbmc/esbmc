# pop() on an empty list raises IndexError inside the list model, which
# checks it with an assert. With IndexError unchecked the path ends there,
# so the ZeroDivisionError reachable only after it is never raised.
def main() -> None:
    l = [1]
    if nondet_bool():
        l.pop()
    n = len(l)
    x = l.pop()
    if n == 0:
        y = 10 // 0


main()
