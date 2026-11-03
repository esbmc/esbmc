class C:
    pass


o = C()
o.b = "x"
assert o.b == "x"
o.b = "abc"
assert len(o.b) == 3
assert o.b.upper() == "ABC"
