class P:
    def __init__(self):
        self.x = 1

def make():
    return P()

assert make().x == 2
