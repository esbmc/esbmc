# nondet_bool controls whether a second entry is added;
# popitem() always removes whichever was last inserted
b: bool = nondet_bool()
d: dict[str, int] = {"x": 10}
if b:
    d["y"] = 20
    key, val = d.popitem()
    assert key == "y"
    assert val == 20
else:
    key, val = d.popitem()
    assert key == "x"
    assert val == 10
