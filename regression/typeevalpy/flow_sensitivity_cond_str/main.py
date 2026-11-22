# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/analysis_sensitivities/flow_sensitivity/arithmetic/arithmetic.py
#
# The str reassignment is made conditional on a nondeterministic value, and
# the final result is used afterwards. Sibling of
# typeevalpy_flow_sensitivity_cond_float.py, where the float reassignment is
# the conditional one instead.

# A simple Python program that reuses the same variable name for different types

def main() -> None:
    a = 1
    b = 2
    result = a + b

    a = 1.0
    b = 2.0
    result = a + b

    if nondet_bool():
        a = "Hello"
        b = "World"
        result = a + b

    shout = result.upper()


main()
