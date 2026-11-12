class Op:
    def __init__(self, a):
        self.a = a

    def double(self):
        return self.a + self.a


assert Op("x").double() == "yy"
