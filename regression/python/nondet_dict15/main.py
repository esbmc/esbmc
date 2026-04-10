def test_nondet_dict_for():
    x = nondet_dict(2)
    __ESBMC_assume(len(x) > 0)
    count: int = 0
    for key in x:
        count = count + 1
    assert count > 0

test_nondet_dict_for()
