# A wrong expectation must NOT be spuriously proven by the consteval folder:
# total(5) is 10, not 999. The folder evaluates the comparison to False and
# declines to short-circuit, so the symbolic engine re-checks it and reports
# the violation.
def total(n):
    s = 0
    i = 0
    while i < n:
        s += i
        i += 1
    return s


assert total(5) == 999
