flag = nondet_bool()
x = "ab" if flag else 5
if isinstance(x, str):
    assert x[0] == "a"
    assert x[1] == "b"
    assert x[-1] == "b"
    assert x[-2] == "a"
    assert ord(x[0]) == 97
