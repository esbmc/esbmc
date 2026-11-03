def main():
    xs = [1, 2]
    g = iter(xs)
    a = next(g)
    b = next(g)
    assert a == 1 and b == 2
    try:
        next(g)
        assert False
    except StopIteration:
        pass


main()
