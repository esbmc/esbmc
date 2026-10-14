def grow(v: list[str]) -> None:
    v.append("z")


zs = ["y"]
grow(zs)
assert "-".join(zs) == "y"
