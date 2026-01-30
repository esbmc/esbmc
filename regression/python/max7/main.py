a: list[int] = [1, 4, 2]
b: list[float] = [1.0, 4.5, 2.0]
c: list[str] = ['a', 'z', 'b']

assert max(a) == 4
assert max(b) == 4.5
assert max(c) == 'z'
