class A:
    pass


class B:
    pass


b = B()
assert type(2.5) is [int, float][1]
assert type("s") is (int, float, str)[-1]
assert type(b) is [A, B][1]
assert type(b) is not [A, B][0]
