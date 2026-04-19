d: dict[str, int] = {"a": 1, "b": 2, "c": 3}

assert 'a' in d, "a is in d"
assert 'b' in d, "b is in d"
assert 'c' in d, "c is in d"
x:int = d['a']
y:int = d['b']
z:int = d['c']
assert x + y + z == 6, "sum of a, b, c is 6"
