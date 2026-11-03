# str.join over the concatenation of two list parameters (#9209).
def g(b: list[str], c: list[str]) -> str:
    return ".".join(b + c)


def h(b: list[str], c: list[str]) -> str:
    return "".join(b + c)


assert g(["a"], ["b"]) == "a.b"
assert g([], ["b"]) == "b"
assert h(["a"], ["b"]) == "ab"
