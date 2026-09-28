# Python evaluates a default once, at definition time, and shares the object
# across calls -- the mutable-default gotcha. The default is hoisted to a
# def-time variable rather than rebuilt per call site, so the accumulation
# below is observable exactly as CPython reports it.


def collect(x, acc=[]):
    acc.append(x)
    return len(acc)


assert collect(1) == 1
assert collect(2) == 2
assert collect(3) == 3

# An explicitly-passed container is independent of the shared default.
assert collect(9, [7]) == 2
assert collect(4) == 4
