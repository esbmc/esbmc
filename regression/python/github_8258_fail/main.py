def add(a: int, b: int, c: int = 10) -> int:
    return a + b + c


xs = [1, 2]
assert add(*xs) == 3
