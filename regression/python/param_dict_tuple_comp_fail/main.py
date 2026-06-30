def f(edges):
    nodes = {v: 0 for u, v in edges}
    total = 0
    for (u, v), w in edges.items():
        total += u + v + w
    return len(nodes) + total


assert f({(0, 1): 10, (1, 2): 20}) == 37
