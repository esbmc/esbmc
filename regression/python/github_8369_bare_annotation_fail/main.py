# A bare annotation binds nothing, so the read raises NameError in
# CPython (#8369).
def f() -> int:
    return d


d: int | str
assert f() == 0
