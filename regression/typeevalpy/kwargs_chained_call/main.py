# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/kwargs/chained_call/main.py
#
# The keyword argument `b` is chosen nondeterministically between func3 and
# func2, and the result is used afterwards.

# Define function with defaults being other functions.
# Then call these functions, passing different functions as keyword arguments.

def func3():
    return "Hello from func3"


def func2(a=func3):
    return a()


def func1(a, b=func2):
    return a(b)


def main() -> None:
    if nondet_bool():
        b = func3
    else:
        b = func2
    c = func1(a=func2, b=b)
    d = c.upper()


main()
