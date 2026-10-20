# A while test runs its str % call operand on every iteration, which a
# hoisted temporary cannot express, so the operand stays refused.
n = 0


def f() -> int:
    global n
    n += 1
    return n


while "%d" % f() != "3":
    pass
assert n == 3
