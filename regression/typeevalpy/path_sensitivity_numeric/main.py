# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/analysis_sensitivities/path_sensitivity/arithmetic/arithmetic.py
#
# The constant branch condition `if True:` is replaced by a nondeterministic
# one, so both branches are feasible, and the result is used afterwards.

# A simple Python program that reuses the same variable name for different types

def main() -> None:
    if nondet_bool():
        a = 1
        b = 2
        temp = a + b
    else:
        a = 1.0
        b = 2.0
        temp = a + b

    result = a + b
    half = result / 2


main()
