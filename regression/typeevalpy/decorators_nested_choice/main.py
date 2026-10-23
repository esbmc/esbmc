# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/decorators/nested_decorators/main.py
#
# A second function with only the outer decorator is added; the function
# called is chosen nondeterministically and its result is used afterwards.

# A function has two decorators, meaning that the first calls the second and the second calls the function.

def dec1(f):
    def inner():
        return f()

    return inner


def dec2(f):
    def inner():
        return 42

    return inner


@dec1
@dec2
def func():
    return "Hello from func"


@dec1
def func_outer_only():
    return "Hello from func"


def main() -> None:
    if nondet_bool():
        f = func
    else:
        f = func_outer_only
    a = f()
    b = a + 1


main()
