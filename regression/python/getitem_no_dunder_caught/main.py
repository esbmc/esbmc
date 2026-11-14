class P:
    def __init__(self):
        self.x = 1


ps = [P(), P()]
caught = False
try:
    v = ps[0][0]
except TypeError:
    caught = True
assert caught
