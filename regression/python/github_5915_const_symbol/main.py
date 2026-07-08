# #5915 follow-up: a constant-valued symbol used as a power base stays exact even
# though the frontend no longer folds symbols -- symex concretises b and d, so the
# multiplication tree yields the exact integer result.
b = 3
d = 4
assert b ** 2 == 9
assert (b + d) ** 2 == 49
assert (d - b) ** 3 == 1
