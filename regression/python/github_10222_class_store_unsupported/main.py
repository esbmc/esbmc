# Issue #10222: a store through a class attribute in the class body was
# dropped.
class A:
    d = {}
    d[1] = 2


assert A.d[1] == 2
