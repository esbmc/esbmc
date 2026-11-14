class C:
    pass


d = {}
d["a"] = C()
assert isinstance(d["a"], int)
