class A:
    def __init__(self):
        self.v = 7

    @classmethod
    def make(cls):
        return cls()


def make(cls):
    return 5


assert A.make().v == 7
assert make(A) == 5
