def grow(v: list[str]) -> None:
    v.append("z")


ws = ["w"]
ws += ["x"]
assert "-".join(ws) == "w-x"

zs = ["y"]
grow(zs)
assert "-".join(zs) == "y-z"
