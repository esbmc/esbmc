# A function nested in a method takes its argument types from the method (#7947).
v = 1


class K:
    def m(self) -> float:
        v = 2.5

        def inner(a):
            return a * 2

        return inner(v)


k = K()
r = k.m()
assert r == 4.0
