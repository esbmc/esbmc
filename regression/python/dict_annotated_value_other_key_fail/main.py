d: dict[int, int] = {}
d[1] = 3
d[2] = "s"
z: int = d[1]
assert z * 2 == 7
