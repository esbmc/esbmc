# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/lists/slice/main.py
#
# The index into the slice is chosen nondeterministically between 1 and 2.

# A new list is created as a slice of another one containing functions.

def func1():
    return 42


def func2():
    return 42.5


def func3():
    return "Hello from func3"


def main() -> None:
    ls = [func1, func2, func3]
    ls2 = ls[1:3]
    if nondet_bool():
        i = 1
    else:
        i = 2
    c = ls2[i]()


main()
