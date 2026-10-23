# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/assignments/starred/main.py
#
# The index into the starred target is chosen nondeterministically between
# 0 and 1, and the returned value is used afterwards.

# Functions are assigned to variables via starred assignment

def func1():
    return "Hello from func1"


def func2():
    return 42


def func3():
    return 42.5


def func4():
    return [2, 4]


def main() -> None:
    a, *b, c = func1, func2, func3, func4
    if nondet_bool():
        i = 0
    else:
        i = 1
    e = b[i]()
    h = e / 2


main()
