# Issue #9388: str.split("") raises ValueError: empty separator.
s: str = "a b"
ok: bool = False
try:
    parts: list[str] = s.split("")
except ValueError:
    ok = True
assert ok

n: int = 0
try:
    n = len(s.rsplit(sep=""))
except ValueError:
    n = -1
assert n == -1

assert len(s.split()) == 2
assert len(s.split(None)) == 2
assert "a,b".split(",")[1] == "b"
