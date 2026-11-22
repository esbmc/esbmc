# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/classes/base_class_calls_child/main.py
#
# The receiver of the inherited func() is chosen nondeterministically
# between the two subclasses, and the returned value is used afterwards.

# The base class calls a function defined by the child class.

class A:
    def func(self):
        return self.child()


class B(A):
    def __init__(self):
        self.child = self.func2

    def func2(self):
        return "Hello from class B"


class C(A):
    def __init__(self):
        self.child = self.func2

    def func2(self):
        return 42


def main() -> None:
    if nondet_bool():
        obj = B()
    else:
        obj = C()
    e = obj.func()
    f = e + 1


main()
