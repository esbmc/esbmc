class C:
    v = 0

    def __init__(self) -> None:
        self.w = 1


a = C()
if nondet_bool():
    a.v = 5
assert a.v == 0 or a.v == 5
