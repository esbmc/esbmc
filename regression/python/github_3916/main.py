def f(a: str) -> str:
    if not a:
        return 'x'
    return a[0]


def last(a: str) -> str:
    return a[-1]


assert f("ab") == "a"
assert f("") == "x"
assert last("ab") == "b"
