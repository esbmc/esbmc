def f() -> None:
    d = {}
    d["p"] = []
    d["p"].append(3)
    assert len(d["p"]) == 2

f()
