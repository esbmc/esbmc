def test_fact(n: int):
    assert n >= 0
    if n == 0:
        return 1
    return n * test_fact(n - 1)

test_fact(-1)
