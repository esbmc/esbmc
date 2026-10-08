# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/args/default/main.py
#
# The call either passes the argument or relies on the default,
# nondeterministically, and the result is used afterwards.

# A function func is defined which takes as a parameter a function which it later calls. It also has a default value assigned
# The 'param_func' function returns a string value.
# The 'param_func2' function returns an integer value.

def param_func():
    return "Hello from param_func"


def param_func2():
    return 1


def func(a=param_func2):
    return a()


def main() -> None:
    if nondet_bool():
        x = func(param_func)
    else:
        x = func()
    y = x + 1


main()
