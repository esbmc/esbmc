# Companion to github_4998 exercising `and` (the other short-circuit operator):
# check('x') is True, so `check(s) and check(s)` is True and f returns 'Yes'.
# The assertion compares against 'No', so it must fail (VERIFICATION FAILED).
# Under the #4998 bug this segfaulted instead of producing a verdict.


def check(t: str) -> bool:
    if len(t) > 0:
        return True
    return False


def f(s: str) -> str:
    return 'Yes' if check(s) and check(s) else 'No'


assert f('x') == 'No'
