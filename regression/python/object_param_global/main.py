class Box:
    value: int = 0


def store(b: Box, v: int) -> Box:
    b.value = v
    return b


shared: Box = Box()
returned = store(shared, 7)

# Reference semantics: the callee mutates the same object the caller holds.
assert returned.value == 7
assert shared.value == 7
