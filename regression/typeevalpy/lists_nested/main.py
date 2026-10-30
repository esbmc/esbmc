# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/lists/nested/main.py
#
# Each branch of a nondeterministic choice calls one of the stored
# functions and uses the returned value.

# Lists containing lists that contain functions.

def func1():
    return 42


def func2():
    return "Hello from func2"


def main() -> None:
    ls = [[func1], func2]
    if nondet_bool():
        c = ls[0][0]() + 1
    else:
        c = ls[1]().upper()


main()
