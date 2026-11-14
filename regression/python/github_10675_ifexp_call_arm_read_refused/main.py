def func():
    return 43


def func_int():
    return 42


flag = nondet_bool()
h = func if flag else func_int()
if not flag:
    assert h + 1 == 43
