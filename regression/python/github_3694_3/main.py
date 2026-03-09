class A:
    def __init__(self):
        self.x = 1

objs = [A()]

assert any(o.x == 1 for o in objs)
