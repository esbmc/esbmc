l = ["a", 1, 2.0]

x: float = l.pop()
assert x == 2.0

y: int = l.pop()
assert y == 1

z: str = l.pop()
assert z == "b"
