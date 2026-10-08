# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/classes/self_assign_func/main.py
#
# The wrapped object is nondeterministically either an A or another B, and
# the returned value is used afterwards.

# A class is assigned as a self item to a class.

class A:
    def func(self):
        return "Hello from classA"


class B:
    def __init__(self, a):
        self.a = a

    def func(self):
        return self.a.func()


def main() -> None:
    if nondet_bool():
        b = B(A())
    else:
        b = B(B(A()))
    c = b.func()
    d = c.upper()


main()
