# `x in obj` / `x not in obj` on a class instance dispatches to obj.__contains__
# (previously a class instance, being a Class* pointer, fell through to the
# string-membership path and gave a wrong result).
class Threshold:
    def __init__(self, limit):
        self.limit = limit

    def __contains__(self, x):
        return x < self.limit


t = Threshold(10)
assert 5 in t
assert 9 in t
assert 10 not in t
assert 20 not in t
assert not (15 in t)


class OnlyFive:
    def __contains__(self, x):
        return x == 5


c = OnlyFive()
assert 5 in c
assert 3 not in c

# Native container membership is unchanged.
assert 2 in [1, 2, 3]
assert "ell" in "hello"
assert 1 in {1: 2}
assert 2 in (1, 2, 3)
assert 9 not in {1, 2, 3}
