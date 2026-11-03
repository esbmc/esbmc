def qualify(modulo: str, name: str) -> str:
    submodulo = f"{modulo}.{name}".lstrip(".")
    return submodulo


x = "a"
s = f"{x}".upper()
assert s == "A"

assert qualify("", "os") == "os"
assert qualify("pkg", "os") == "pkg.os"

n = 0
for c in ["b", "c"]:
    t = f"{c}!".upper()
    assert t == c.upper() + "!"
    n += 1
assert n == 2
