# Simple function returning a constant
def func() -> int:
    return 2

# Function with parameter
def id(x: int) -> int:
    return x

# Nested call
assert id(func()) == 1
