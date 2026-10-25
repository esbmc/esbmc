# Assigning a str method called on an f-string (#9207).
def name(m: str) -> str:
    s = f"{m}".upper()
    return s


x = "a"
s = f"{x}".upper()
assert s == "A"
t = f".{x}.".lstrip(".")
assert t == "a."
ok = f"{x}b".startswith("a")
assert ok
assert name("b") == "B"
for y in ["c", "d"]:
    u = f"{y}".upper()
    assert u == "C" or u == "D"
