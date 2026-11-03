p = [1, 2, 3, 4]
a, *b, c = p
assert a == 1 and c == 4 and len(b) == 2
q = [1]
try:
    a, *b, c = q
    assert False
except ValueError:
    pass
