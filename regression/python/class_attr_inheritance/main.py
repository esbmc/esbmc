class A:
    class_attr = 1

class B(A):
    pass

b = B()
assert b.class_attr == 1
