# keys()/values()/items() are iterated rather than indexed: CPython returns
# views, which are not subscriptable.
d = {"a": 1, "b": 2}
assert d.get("a") == 1
assert d.get("z", -1) == -1

total = 0
for k, v in d.items():
    total = total + v
assert total == 3

seen = 0
for k in d.keys():
    seen = seen + 1
assert seen == 2

summed = 0
for v in d.values():
    summed = summed + v
assert summed == 3

assert d.setdefault("a", 5) == 1
assert d.setdefault("c", 7) == 7
assert d["c"] == 7
assert len(d) == 3

assert d.pop("c") == 7
assert len(d) == 2
assert d.pop("zz", 4) == 4

e = d.copy()
e["x"] = 10
assert len(e) == 3
assert len(d) == 2

d.update({"m": 5})
assert d["m"] == 5
assert len(d) == 3

d.clear()
assert len(d) == 0
