# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/classes/inheritance_overriding/main.py
#
# The receiver is chosen nondeterministically between the base class and
# the subclass, and the returned value is used afterwards.

# Method Overriding by imherited class

class MyClass:
    def func(self):
        return "Hello from func in MyClass"


class MySubClass(MyClass):
    def func(self):
        return 42


def main() -> None:
    MyClass().func()
    if nondet_bool():
        a = MySubClass()
    else:
        a = MyClass()
    b = a.func()
    c = b + 1


main()
