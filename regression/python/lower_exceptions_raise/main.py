# Symbolic exception lowering (#5075) for Python: an inter-procedural raise is
# caught by an except handler binding the exception object.
def foo(value: int) -> int:
    if value < 0:
        raise ValueError("Negative value!")
    return value * 2


result = 1
try:
    result = foo(-1)
except ValueError as e:
    result = 7

assert result == 7
