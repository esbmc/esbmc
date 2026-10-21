def ends(g):
    return {v: 0 for u, v in g}


def weights(g):
    return {v: w for (u, v), w in g.items() if w > 0}


def main() -> None:
    g = {("a", "b"): 3, ("c", "d"): 6, ("e", "f"): -1}
    r = ends(g)
    assert len(r) == 3
    assert r["b"] == 0 and r["d"] == 0 and r["f"] == 0
    assert "a" not in r
    w = weights(g)
    assert len(w) == 2
    assert w["b"] == 3 and w["d"] == 6
    assert "f" not in w


main()
