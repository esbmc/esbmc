# Positive twin of nondet_list25_fail: the same returned-from-a-function
# position, with a property that holds for every expansion.
def make():
    return nondet_list(3)


x = make()
assert len(x) <= 3
