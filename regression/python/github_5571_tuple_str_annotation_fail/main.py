def f(pairs: list[tuple[str, str]]):
    s = ""
    for u, v in pairs:
        s = v
    return s

assert f([("A", "B")]) == "A"
