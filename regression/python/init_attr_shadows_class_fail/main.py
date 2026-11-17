class A:
    y = 0

    def __init__(self, c):
        if c:
            self.y = 5


assert A(False).y == 5
