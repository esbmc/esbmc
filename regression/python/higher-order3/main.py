def make_multiplier(k):
    def mul(x):
        return x * k
    return mul


times3 = make_multiplier(3)
# Once the closure escapes make_multiplier, mul still reads k from the dead
# enclosing frame, so the product is unconstrained (#6256).
assert times3(4) == 12
