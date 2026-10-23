# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/lambdas/chained_calls/main.py
#
# The last lambda is chosen nondeterministically between the original one
# and one returning a str, and the final result is used afterwards.

# Lambdas are passed via parameters to chained functions and then called.

def func3(a):
    return a(1)


def func2(a, b):
    a(1)
    return func3(b)


def func1(a, b, c):
    a(1)
    return func2(b, c)


def main() -> None:
    if nondet_bool():
        last = lambda x: x + 3
    else:
        last = lambda x: str(x)
    d = func1(lambda x: x + 1, lambda x: x + 2, last)
    e = d + 1


main()
