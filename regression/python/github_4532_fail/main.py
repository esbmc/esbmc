class T:
    def __init__(self, d0: int, d1: int):
        self.shape: tuple = (d0, d1)

def f(t: T) -> int:
    A, B = t.shape
    s: int = 1
    for i in range(3):
        x: int = 0
        if i + s < A:
            x = i
        else:
            x = A
        s = s + x
    return s

t: T = T(10, 20)
assert f(t) == 5
