class C:
    pass


x = C()
d = {}
d["a"] = x
assert isinstance(d["a"], C)
