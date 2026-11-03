xs = ["a"]
xs.append("b")


def f() -> str:
    return ",".join(xs)


assert f() == "a,b"
