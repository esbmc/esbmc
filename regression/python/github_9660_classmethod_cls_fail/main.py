class A:
    def __init__(self):
        self.v = 7

    @classmethod
    def make(cls):
        return cls()


assert A.make().v == 8
