# A function that falls off the end returns None (#10112).

def func():
    pass


class C:
    def __init__(self):
        self.x = 1

    def bump(self):
        self.x += 1
