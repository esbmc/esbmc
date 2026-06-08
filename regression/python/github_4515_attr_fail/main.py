class T:
    def __init__(self, d0: int, d1: int):
        self.shape: tuple = (d0, d1)

t: T = T(10, 20)
M, N = t.shape
assert M == 99
