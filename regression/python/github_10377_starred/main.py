class Base:
    def __init__(self):
        self.data = [0, 1, 2, 3]

    def get(self, pair):
        (x, *rest), y = pair
        return self.data[y] + x


class Sub(Base):
    def get(self, idx):
        return self.data[idx]


assert Sub().get(2) == 2
