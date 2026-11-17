f = lambda y: y * 2
for i in range(3):
    r = f(i)
    assert isinstance(r, int)
