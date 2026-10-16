a = "c"
zs = [a, "d"]
assert "".join(zs) == "cd"

ys: list[str] = []
ys.append("p")
ys.append("q")
assert ",".join(ys) == "p,q"
