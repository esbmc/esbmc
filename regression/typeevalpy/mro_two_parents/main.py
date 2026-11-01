# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/mro/two_parents/main.py
#
# The receiver is chosen nondeterministically between the class with two
# parents and its first parent, and the returned value is used afterwards.

# A class is defined with two parents. The correct ordering must be preserved when calling a parent function.

class A:
    def func(self):
        return 42


class B:
    def __init__(self):
        pass

    def func(self):
        return "Hello from class B"


class C(A, B):
    pass


def main() -> None:
    if nondet_bool():
        obj = C()
    else:
        obj = A()
    d = obj.func()
    e = d + 1
    B().func()


main()
