# Calling a function taken out of a dict gives a value of no known type; the
# equality on it was folded to False against the other operand's type, which
# reported a correct program as failing. It is refused now.
def as_str():
    return "a"


def as_int():
    return 1


def main():
    table = {"s": as_str, "i": as_int}
    key = "s" if nondet_bool() else "i"
    c = table[key]()
    if key == "s":
        assert c == "a"


main()
