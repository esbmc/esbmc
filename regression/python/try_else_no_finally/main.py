# `try/except/else` WITHOUT a finally must NOT be refused (the else-clause
# refusal is scoped to try/finally). Here the exception is raised and caught, so
# the else branch is not taken and the program completes normally. Guards
# against re-introducing an over-aggressive else refusal (cf. github_3090).
x = 0
try:
    raise ValueError()
except ValueError:
    x = 1
else:
    x = 2

assert x == 1
