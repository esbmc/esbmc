class C:
    pass


c = C()
c.k = 1
assert getattr(c, "k", 2) == 2
