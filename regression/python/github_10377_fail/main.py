class Base:
    def __init__(self):
        self.data = [0.0, 1.0, 2.0, 3.0]

    def get(self, x_y):
        (x, y) = x_y
        return self.data[y * 2 + x]


class Sub(Base):
    def get(self, idx):
        return self.data[idx[1] * 2 + idx[0]]


assert Sub().get((1, 1)) == 2.0
