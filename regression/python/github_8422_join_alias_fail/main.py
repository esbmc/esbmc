xs = ["a"]
ys = xs
ys.append("c")
assert ",".join(xs) == "a"
