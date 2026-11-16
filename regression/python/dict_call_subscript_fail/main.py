def pairs_to_dict(g):
    r = {v: 0 for u, v in g}
    return r


assert pairs_to_dict({("A", "B"): 3})["B"] == 1
