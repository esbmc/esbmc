x: int = nondet_int()
xs = [1, 2, 3, 2]

if x == 1 or x == 2 or x == 3:
    xs.remove(x)

    if x == 2:
        # remove() deletes only the first matching value.
        assert xs[2] != 2
