# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/classes/inheritance_overriding/main.py
#
# Each branch of a nondeterministic choice calls func() on one of the two
# classes and uses the returned value.

# Method Overriding by imherited class

class MyClass:
    def func(self):
        return "Hello from func in MyClass"


class MySubClass(MyClass):
    def func(self):
        return 42


def main() -> None:
    if nondet_bool():
        c = MySubClass().func() + 1
    else:
        c = MyClass().func().upper()


main()
