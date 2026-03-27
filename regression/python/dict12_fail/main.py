def test_float_values():
    constants: dict = {'pi': 3.14159, 'e': 2.71828, 'phi': 1.61803}

    assert constants['pi'] < 3.0
    assert constants['e'] > 3.0
    assert constants['phi'] < 1.5


test_float_values()
