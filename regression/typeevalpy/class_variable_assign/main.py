# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/classes/class_variable/main.py
#
# Either the class variable or an instance attribute of the same name is
# assigned, nondeterministically, and the attribute is used as a divisor.

# Class Variable is assigned to a variable

class MyClass:
    class_var = 0

    def __init__(self, instance_var):
        self.instance_var = instance_var


def main() -> None:
    a = MyClass(10)
    if nondet_bool():
        MyClass.class_var = 2
    else:
        a.class_var = 5
    ratio = a.instance_var / a.class_var


main()
