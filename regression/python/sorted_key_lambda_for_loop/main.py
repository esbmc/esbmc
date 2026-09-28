def run():
    xs = [3, 1, 2]
    xs.append(4)
    seen = 0
    # The mutated list is not constant-foldable, so this reaches the scan.
    for v in sorted(xs, key=lambda a: -a):
        if seen == 0:
            assert v == 4
        seen = seen + 1
    assert seen == 4


run()
