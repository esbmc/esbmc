class C:
    pass


o = C()
o.b = "x"
o.b = "abc"
assert o.b[1] == "a"
