x = 1
y = "s"
d = {}
for i in range(2):
    if i == 0:
        d["a"] = x
    else:
        d["a"] = y
    assert isinstance(d["a"], int)
