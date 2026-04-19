def test_bool_nondet_list():
    x = nondet_list(8, nondet_bool())
    # x is a list with nondet length [0, 8] and nondet int elements
    assert len(x) >= 0

test_bool_nondet_list()

def test_int_nondet_list():
    x = nondet_list(8, nondet_int())
    # x is a list with nondet length [0, 8] and nondet int elements
    assert len(x) >= 0

test_int_nondet_list()

def test_float_nondet_list():
    x = nondet_list(8, nondet_float())
    # x is a list with nondet length [0, 8] and nondet int elements
    assert len(x) >= 0

test_float_nondet_list()
