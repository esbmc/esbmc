def f(edges):
    # dict comprehension unpacking the tuple keys of a parameter dict
    nodes = {v: 0 for u, v in edges}
    # nested-tuple target over the same parameter dict's items
    total = 0
    for (u, v), w in edges.items():
        total += u + v + w
    return len(nodes) + total


assert f({(0, 1): 10, (1, 2): 20}) == 36
