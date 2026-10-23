# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/analysis_sensitivities/context_sensitivity/arithmetic/arithmetic.py
#
# The value used afterwards is chosen nondeterministically between the
# results of the int and str calls.

# A simple Python program that defines a function called 'arithmetic_op'.
# The function takes two integer parameters 'a' and 'b', adds them together  and returns result.
# The given code is context sensitive because it produces different results based on the context in which it is executed.

def arithmetic_op(a, b):
    result = a + b
    return result


def main() -> None:
    result1 = arithmetic_op(5, 10)
    result2 = arithmetic_op(5.2, 10.3)
    result3 = arithmetic_op("Hello", "World")

    if nondet_bool():
        x = result1
    else:
        x = result3
    half = x / 2


main()
