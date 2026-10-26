def qualify(modulo: str, name: str) -> str:
    submodulo = f"{modulo}.{name}".lstrip(".")
    return submodulo


def shout(m: str) -> str:
    s = f"{m}".upper()
    return s


x = "a"
s = f"{x}".upper()
assert s == "A"
t = f".{x}.".lstrip(".")
assert t == "a."
ok = f"{x}b".startswith("a")
assert ok
assert shout("b") == "B"

assert qualify("", "os") == "os"
assert qualify("pkg", "os") == "pkg.os"

n = 0
for c in ["b", "c"]:
    t = f"{c}!".upper()
    assert t == c.upper() + "!"
    n += 1
assert n == 2
