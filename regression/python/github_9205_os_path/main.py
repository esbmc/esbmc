import os


def f(a: str, b: str) -> str:
    return os.path.join(a, b)


assert f("x", "y") == "x/y"
assert f("x/", "y") == "x/y"
assert f("x", "/y") == "/y"
assert f("", "y") == "y"

r: str = os.path.realpath("x")
s: str = os.path.relpath("/a/b", "/a")
c: str = os.path.commonpath(["/a/b", "/a/c"])
if os.path.islink("x"):
    try:
        os.unlink("x")
    except FileNotFoundError:
        pass
