# `sorted(d)` over a tuple-keyed dict lowers to `sorted(d.keys())`, whose
# argument is a call rather than a named list, so the convert-time tuple fold
# declined and the generic path retyped the keys as int -- a later
# `u, v = edge` could not unpack them.


def main() -> None:
    d = {(3, 1): 10, (1, 2): 20, (2, 9): 30}

    order = []
    for edge in sorted(d):
        u, v = edge
        order.append(u)

    # sorted(), not insertion order.
    assert order[0] == 1
    assert order[1] == 2
    assert order[2] == 3

    # The same keys reached through an explicit list keep working.
    ks = list(d.keys())
    for edge in sorted(ks):
        a, b = edge
        assert a <= 3

    # Plain iteration is unchanged.
    for edge in d:
        a, b = edge
        assert b >= 1


main()
