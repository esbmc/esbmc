x = nondet_int()
use_str = nondet_bool()


def nested_func():
    return "Hello from nested_func"


def nested_func2():
    return x


def param_func(a):
    return a()


def func(a):
    if use_str:
        return a(nested_func)
    return a(nested_func2)


b = param_func
c = func
d = c(b)

if use_str:
    assert isinstance(d, str)
    assert d == "Hello from nested_func"
else:
    assert isinstance(d, int)
    assert d == x
