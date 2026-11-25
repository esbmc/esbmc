def func():
    return "a"


flag = nondet_bool()
h = func if flag else 42
if not flag:
    assert h == 42
