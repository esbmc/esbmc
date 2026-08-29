def make_multiplier(k):
    def mul(x):
        return x * k
    return mul


times3 = make_multiplier(3)
# Two calls through the same escaped closure, both reading k == 3 (#6256).
assert times3(4) == 12
assert times3(5) == 15
