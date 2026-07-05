# Pins the single-write guard in the consteval global seeder. At the assert,
# N == 3, so f(N) == 100 is False and the program must report VERIFICATION
# FAILED. N is reassigned to 100 later in the module; if the seeder were to
# fold the assert with that stale final value it would compute f(100) == 100
# (True) and wrongly prove the assertion. The guard refuses to seed any global
# written more than once, so the assert falls through to the solver and fails.
def f(x):
    s = 0
    i = 0
    while i < x:
        s += 1
        i += 1
    return s


N = 3
assert f(N) == 100
N = 100
