def run(a: int):
    xs = [a, a + 1]
    assert max(xs, key=lambda v: -v) == a
    assert min(xs, key=lambda v: -v) == a + 1


run(3)
