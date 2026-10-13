# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/analysis_sensitivities/inter_procedural/arithmetic/arithmetic.py
#
# The first argument is chosen nondeterministically between 5 and -5, and
# the result is used as a divisor afterwards.

# A simple Python program that defines a function called 'arithmetic_op' which calls another function 'add' to perform addition of two integer parameters 'a' and 'b'.
# The 'add' function takes two integer parameters 'a' and 'b', adds them together, and returns the result.
# The given code is interprocedural because it involves calling a separate function ('add') to complete the arithmetic operation.

def arithmetic_op(a, b):
    result = add(a, b)
    return result


def add(a, b):
    result = a + b
    return result


def main() -> None:
    a = 5
    if nondet_bool():
        a = -5
    result = arithmetic_op(a, 10)
    ratio = 100 / result


main()
