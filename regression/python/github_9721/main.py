def b(x: str):
    return object()


s = b("str")
assert isinstance(s, object)
assert isinstance(3, object)
assert isinstance("a", object)
assert isinstance([1], (str, object))
