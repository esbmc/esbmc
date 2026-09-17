x: int = nondet_int()
xs = [1, 2, 3, 2]

if x == 1 or x == 2 or x == 3:
    xs.remove(x)
    assert len(xs) == 3

    if x == 1:
        assert xs[0] == 2
        assert xs[1] == 3
        assert xs[2] == 2
    elif x == 2:
        assert xs[0] == 1
        assert xs[1] == 3
        assert xs[2] == 2
    else:
        assert xs[0] == 1
        assert xs[1] == 2
        assert xs[2] == 2
