# Issue #9624: x = x.method() over a list parameter passed &x as self
# after x was retyped from the struct to the reference.
class Point:
    def __init__(self, i):
        self.x = i * 1.5

    def maximize(self, other):
        if self.x < other.x:
            self.x = other.x
        return self


def maximize(points):
    nxt = points[0]
    for p in points[1:]:
        nxt = nxt.maximize(p)
    return nxt


pts = [Point(0), Point(2)]
r = maximize(pts)
assert r.x == 3.0
assert pts[0].x == 3.0
