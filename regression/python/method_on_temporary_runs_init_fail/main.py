# The receiver's state is genuinely constrained by __init__, not merely
# unchecked: asserting the wrong value must be refutable.


class C:
    def __init__(self):
        self.n = 3

    def size(self):
        return self.n


assert C().size() == 4
