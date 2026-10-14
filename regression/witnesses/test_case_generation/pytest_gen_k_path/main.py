# Regression for issue #4337: --k-path-coverage drives
# --generate-pytest-testcase end-to-end on a Python program with
# correlated branches. Exercises one witness per (prior-direction,
# current-direction) combination rather than only per branch direction.


def f(a: int, b: int) -> int:
    x = 0
    if a > 0:
        x += 1
    if b > 0:
        x += 1
    return x


f(__VERIFIER_nondet_int(), __VERIFIER_nondet_int())
