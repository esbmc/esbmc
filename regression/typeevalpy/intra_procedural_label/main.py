# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/analysis_sensitivities/intra_procedural/arithmetic/arithmetic.py
#
# Operand `b` is conditionally set to -1, and the result is used as a key
# into a table of labels afterwards.

# This program is an example of intraprocedural analysis.
# The function arithmetic_op takes two integer parameters a and b, adds them together, and returns the result as a string indicating whether the result is "Positive" or "Negative" based on whether it is greater than or less than zero.
# The program does not call any other functions, so the analysis is focused on the behavior of this single function.

def main() -> None:
    a = 1
    b = 2
    if nondet_bool():
        b = -1
    result = a + b

    labels = {3: "Positive", -3: "Negative"}
    label = labels[result]


main()
