# Calling a parameter named like the class passed for it (#9806).
class A:
    def __init__(self):
        pass


def f(A):
    x = A()


f(A)
