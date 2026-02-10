import math


POINTS = 100000


class Point(object):

    def __init__(self, i):
        self.x = math.sin(i)

    def normalize(self):
        x = self.x / 100

n = POINTS

points = [Point(0)] * n

for i in range(n):
    points[i] = Point(i)

for p in points:
    p.normalize()
