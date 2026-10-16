# pop() on a list that may be empty and is not inside a try reports the
# IndexError as uncaught (#8504).
def main() -> None:
    s: list[int] = []
    if nondet_bool():
        s.append(1)
    x = s.pop()
    assert x == 1


main()
