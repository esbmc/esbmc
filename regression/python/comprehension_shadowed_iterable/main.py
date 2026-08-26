# Python evaluates a comprehension's leftmost iterable in the enclosing scope,
# before the comprehension binds its target. Lowering it inside the generated
# loop evaluated it after, so the iterable read the rebound element instead.


def main():
    xs = [1, 2, 3]
    node = xs
    out = [node for node in node]
    assert len(out) == 3
    assert (out[0]) == 1
    assert (out[2]) == 3

    ys = [4, 5]
    item = ys
    doubled = [item * 2 for item in item]
    assert len(doubled) == 2
    assert (doubled[1]) == 10


main()
