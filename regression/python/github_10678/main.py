# t is a function only when use_str holds, and is only called then.
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
if use_str:
    b = t()
    assert b == "Hello from return_func"
