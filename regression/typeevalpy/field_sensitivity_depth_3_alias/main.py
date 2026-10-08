# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/analysis_sensitivities/field_sensitivity_depth_3/depth_3/depth_3.py
#
# The innermost object is conditionally replaced by the middle one before
# compute().

# Program for field sensitivity analysis in depth 3

class ArithmeticOperation:
    def __init__(self, a, b):
        self.a = a
        self.result = None
        self.nested = self.Nested(b)

    class Nested:
        def __init__(self, b):
            self.nested2 = self.Nested2(b)

        class Nested2:
            def __init__(self, c):
                self.c = c

    def compute(self):
        self.result = self.a + self.nested.nested2.c
        return self.result


def main() -> None:
    arith_op = ArithmeticOperation(5, 4)
    if nondet_bool():
        arith_op.nested.nested2 = arith_op.nested
    result1 = arith_op.compute()


main()
