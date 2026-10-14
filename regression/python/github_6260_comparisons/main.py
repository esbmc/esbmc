# Identity and equality stay defined on None; only arithmetic and ordering
# raise (GitHub #6260).
x = None
assert x is None
assert not (x is not None)
assert x == None
assert not (x != None)

y = 5
assert y is not None
assert not (y is None)


def maybe(a: int):
    if a > 0:
        return a
    return None


assert maybe(-1) is None
assert maybe(2) is not None
