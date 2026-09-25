# Negative counterpart: the same call-return binding with a false assertion. The
# converted result must reach the solver and report the violation.
def count_up(xs: list) -> int:
    n = len(xs)
    i = 0
    while i < n:
        i = i + 1
    return i


assert count_up([1, 2, 3]) == 99
