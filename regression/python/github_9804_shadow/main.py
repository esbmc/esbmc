import helper


def func():
    return 42


g = helper.func
h = func
assert g() == 41
assert h() == 42
