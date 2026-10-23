d: dict[int, int] = {}
d[1] = 3
k = 2
d[k] = "s"
i = 1
z: int = d[i]
assert z * 2 == 6
w: int = d[1]
assert w * 2 == 6
