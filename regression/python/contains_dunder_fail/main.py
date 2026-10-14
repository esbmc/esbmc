# __contains__ returns x == 5, so 3 in c is False.
class C:
    def __contains__(self, x):
        return x == 5


c = C()
assert 3 in c
