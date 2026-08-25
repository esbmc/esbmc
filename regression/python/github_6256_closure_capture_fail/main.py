# The refutable half of #6256: the captured n is bound to 5, so add5(3) is 8
# and the claim below is false. A capture cell that stayed unconstrained would
# also report this, so github_6256_closure_capture is what pins the fix.


def make_adder(n):
    def add(x):
        return x + n

    return add


add5 = make_adder(5)
assert add5(3) == 9
