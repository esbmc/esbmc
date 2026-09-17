def make_multiplier(k):
    def mul(x):
        return x * k
    return mul


times3 = make_multiplier(3)
# The closure escapes make_multiplier and still reads k == 3 (#6256).
assert times3(4) == 12
