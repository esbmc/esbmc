l: list[int] = [1, 2, 3]
b: bool = not all(x > 0 for x in l)
assert b == False

if b:
    assert False

l2: list[int] = [1, -1, 3]
b2: bool = not all(x > 0 for x in l2)
assert b2 == True

if b2:
    pass
else:
    assert False
