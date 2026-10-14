class Box:
    value: int = 0


def store(b: Box, v: int) -> Box:
    b.value = v
    return b


shared: Box = Box()
store(shared, 7)

# Wrong: the mutation through the parameter IS visible on the caller's object,
# so shared.value is 7, not 0.
assert shared.value == 0
