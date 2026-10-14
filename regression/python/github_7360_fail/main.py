def run():
    xs = [[3, 1], [1, 9, 9]]
    xs[0] = xs[1]
    # Wrong on purpose: 2 is the length of the element that was overwritten,
    # which is exactly what ESBMC used to answer here.
    assert len(xs[0]) == 2


run()
