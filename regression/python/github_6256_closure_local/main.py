# A closure over an enclosing *local* (not a parameter): peek escapes
# make_counter and must still read start == 41 (#6256).


def make_counter():
    start = 41

    def peek():
        return start + 1

    return peek


nxt = make_counter()
assert nxt() == 42
