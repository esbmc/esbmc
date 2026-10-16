# A caught exception does not end the path, checked or not, so the IndexError
# in the handler is found.
def main() -> None:
    b = nondet_int()
    l = [1, 2]
    try:
        q = 10 // b
    except ZeroDivisionError:
        x = l[2]


main()
