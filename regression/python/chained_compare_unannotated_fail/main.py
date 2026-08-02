# 20 is not within [1, 10], so the chained comparison is False.
class Range:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def check(self, x):
        return self.lo <= x <= self.hi


assert Range(1, 10).check(20)
