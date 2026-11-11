# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/returns/multiple_types/main.py
#
# The argument is chosen nondeterministically between 5 and -5, and the
# returned value is used afterwards.

# Returning multiple types from a function using type hinting.

def func(x):
    if x > 0:
        return x
    else:
        return "Invalid input"


def main() -> None:
    if nondet_bool():
        x = 5
    else:
        x = -5
    a = func(x)
    b = a + 1


main()
