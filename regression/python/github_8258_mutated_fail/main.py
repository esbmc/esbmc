def add(a: int, b: int) -> int:
    return a + b


xs = [1, 2]
xs.append(3)
assert add(*xs) == 3
