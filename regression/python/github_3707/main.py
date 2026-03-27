def f(x: int, y: int) -> int:
    return x * y

g = f

def h(op=g):
    return op(1, 1)

result = h()
assert result == 1
