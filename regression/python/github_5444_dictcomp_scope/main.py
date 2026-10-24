def ends(g):
    v = 5
    d = {v: 0 for u, v in g}
    assert v == 5
    assert len({v: 0 for u, v in g}) == 1
    assert v == 5
    return d


d = {"a": 1, "b": 2}
k = "z"
r = {k: v for k, v in d.items() if v > 1}
assert k == "z" and len(r) == 1 and r["b"] == 2
e = ends({(1, 2): 3})
assert e[2] == 0

c = {1: 0, 2: 0}
n = {k: sum([k for k in [5, 6]]) + (lambda j: j + k)(10) for k in c}
assert n[1] == 22 and n[2] == 23
xs = [1, 2]
p = {a: b for a, b in [(x, x + 1) for x in xs]}
assert p[1] == 2
