def test_extend_symbolic_string(s: str):
    a = ['x']
    a.extend(s)
    assert len(a) >= 1

x = nondet_str()
test_extend_symbolic_string(x)
