def f(a: str, b: str) -> str:
    return "".join([a, b])


assert f("p", "q") == "pq"
assert f("", "q") == "q"
