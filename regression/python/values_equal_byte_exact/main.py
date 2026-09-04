# __ESBMC_values_equal's 8-byte fast path reads all eight bytes, so every byte
# has to count: `high` and 0 agree on all but the most significant one.
def main():
    xs = [1, 2, 3]
    assert 2 in xs
    assert 4 not in xs

    high = 1 << 56
    ys = [0]
    assert high not in ys

    zs = [high]
    assert 0 not in zs
    assert high in zs


main()
