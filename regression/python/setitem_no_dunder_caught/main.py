class P:
    def __init__(self):
        self.x = 1


ps = [P(), P()]
caught = False
try:
    ps[0][0] = 3
except TypeError:
    caught = True
assert caught
