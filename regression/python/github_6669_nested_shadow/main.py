count: int = 0


def outer() -> None:
    global count

    def helper() -> int:
        # A `global` in the nested scope must not leak out to the enclosing
        # one, and must not survive past this def either.
        global count
        count = count + 10
        return count

    count = 1
    assert helper() == 11
    count = count + 1


outer()
assert count == 12
