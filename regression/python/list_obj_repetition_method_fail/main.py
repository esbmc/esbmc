# Negative variant: the method resolves and returns the real per-element
# value, so an assertion on the wrong total must fail.
class Point:
    def __init__(self, i):
        self.x = i

    def value(self):
        return self.x * 2


n = 3
pts = [Point(0)] * n
for i in range(n):
    pts[i] = Point(i)

total = 0
for p in pts:
    total += p.value()

assert total == 7  # wrong: real total is (0 + 1 + 2) * 2 == 6
