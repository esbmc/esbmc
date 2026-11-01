# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/analysis_sensitivities/path_sensitivity/arithmetic/arithmetic.py
#
# The constant branch condition `if True:` is replaced by a nondeterministic
# one, so both branches are feasible, and the result is used afterwards.
# Unlike typeevalpy_path_sensitivity_numeric.py, the second branch uses str
# operands instead of float ones.

# A simple Python program that reuses the same variable name for different types

def main() -> None:
    if nondet_bool():
        a = 1
        b = 2
        temp = a + b
    else:
        a = "Hello"
        b = "World"
        temp = a + b

    result = a + b
    half = result / 2


main()
