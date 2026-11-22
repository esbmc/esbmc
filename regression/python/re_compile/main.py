import re

p = re.compile(r"b")
assert p.search("abc")
assert p.match("bcd")

q = re.compile(r"b", re.IGNORECASE)
assert q.search("aBc")

w = re.compile("[a-z]+")
assert w.fullmatch("abc")
assert not w.fullmatch("ab1")


def check(s: str) -> bool:
    return w.fullmatch(s)


assert check("xyz")
