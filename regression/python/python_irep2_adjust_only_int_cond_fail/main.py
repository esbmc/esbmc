# Negative counterpart: the same integer-valued branch and loop conditions with
# a false assertion. The cast must let the guard reach the solver and report the
# violation, not abort the encoding.
def classify(n: int) -> int:
    total = 0
    if n:
        total = 1

    i = 2
    while i:
        total = total + 1
        i = i - 1

    return total


assert classify(5) == 99
