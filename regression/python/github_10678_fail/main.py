# t() calls 42 when use_str is false, which raises TypeError.
use_str = nondet_bool()


def return_func():
    return "Hello from return_func"


def return_func2():
    return 42


def func():
    if use_str:
        a = return_func
    else:
        a = return_func2()
    return a


t = func()
b = t()

if use_str:
    assert b == "Hello from return_func"
else:
    assert b == 42
