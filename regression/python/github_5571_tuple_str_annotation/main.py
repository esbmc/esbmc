def f(pairs: list[tuple[str, str]]):
    s = ""
    for u, v in pairs:
        s = v
    return s

# Two direct call sites with the same literal shape: the call-site sizing
# only fires when every site's recovered annotation agrees (ast.dump
# equality), so this pins the cross-site agreement path.
assert f([("A", "B")]) == "B"
r = f([("A", "B")])
assert r == "B"
