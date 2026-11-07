# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/analysis_sensitivities/field_sensitivity_depth_2/arithmetic_depth_2/arithmetic_depth_2.py
#
# The value stored in the nested object is chosen nondeterministically
# between an int and a float, and the result is used afterwards.

# The given code is an example of field sensitivity because it can recognise values based on the values assigned to its member variables.
# Also it has multiple depth in field sensitivity

class ArithmeticOperation:
    def __init__(self, a, b):
        self.a = a
        self.result = None
        self.nested = self.Nested(b)

    class Nested:
        def __init__(self, b):
            self.b = b

    def compute(self):
        self.result = self.a + self.nested.b
        return self.result


def main() -> None:
    if nondet_bool():
        b = 4
    else:
        b = 4.5
    arith_op = ArithmeticOperation(5, b)
    result1 = arith_op.compute()
    half = result1 / 2


main()
