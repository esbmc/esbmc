# Simple function returning a constant
def func() -> int:
    return 2


# Function with parameter
def id(x: int) -> int:
    return x


# Recursive (bounded) factorial
def fact(n: int) -> int:
    if n <= 1:
        return 1
    return n * fact(n - 1)


# Conditional return
def threshold(x: int) -> int:
    if x > 10:
        return 100
    return 1


# Call func, check nonzero
x: int = func()
y: float = 1.0 / x
assert y >= 0.5  # should hold

# Call with param
assert id(3) == 3
assert id(0) == 0

# Recursive function (only up to 3 to keep symbolic depth shallow)
assert fact(3) == 6
assert fact(0) == 1

# Conditional return
assert threshold(11) == 100
assert threshold(5) == 1
