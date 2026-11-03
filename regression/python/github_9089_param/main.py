def measure(s: str) -> tuple[str, int]:
    return (s, len(s))


name, n = measure("abc")
assert name == "abc"
assert n == 3
empty, m = measure("")
assert empty == ""
assert m == 0
