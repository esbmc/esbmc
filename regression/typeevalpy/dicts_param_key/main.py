# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/dicts/param_key/main.py
#
# The key passed in the second call is chosen nondeterministically between
# "b" and "c".

# The key of a dictionary is passed as a function parameter.

def func1(key="a"):
    return d[key]()


def func2():
    return "Hello from func2"


def func3():
    return 42


d = {"a": func2, "b": func3}


def main() -> None:
    e = func1()
    if nondet_bool():
        key = "b"
    else:
        key = "c"
    f = func1(key)


main()
