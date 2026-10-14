def summarize(g):
    n = 0
    for k in g:
        n += 1
    s = 0
    for k, w in g.items():
        s += w
    return n + s


assert summarize({'a': 3, 'b': 7}) == 12
