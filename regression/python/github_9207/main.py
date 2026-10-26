# A str method called on an f-string and assigned (#9207).
x = "a"
s = f"{x}".upper()
assert s == "A"
t = f".{x}".lstrip(".")
assert t == "a"


def f(y: str) -> str:
    r = f"{y}!".upper()
    return r


assert f("q") == "Q!"
