def weights(g):
    return {v: w for (u, v), w in g.items() if w > 0}


def main() -> None:
    g = {("a", "b"): 3, ("c", "d"): 6, ("e", "f"): -1}
    w = weights(g)
    assert w["d"] == 3


main()
