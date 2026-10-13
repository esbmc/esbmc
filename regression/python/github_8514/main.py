class Bag:

    def __init__(self):
        self.n = 2

    def count(self):
        return self.n


assert Bag().count() == 2
