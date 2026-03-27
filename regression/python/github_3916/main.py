def f(a: str) -> str:
    if not a:
        return 'x'
    return a[0]


assert f("ab") == "a"
assert f("") == "x"
