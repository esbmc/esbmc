import mod

for i in range(2):
    a = mod.A()
    if not isinstance(a.b, str):
        raise TypeError("lost")
    if i == 1:
        assert a.b == "x"
