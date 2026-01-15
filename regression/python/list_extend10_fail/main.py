def test_extend_symbolic_string(s: str):
    a = ['x']
    a.extend(s)
    assert len(a) >= 4

test_extend_symbolic_string("ab")
