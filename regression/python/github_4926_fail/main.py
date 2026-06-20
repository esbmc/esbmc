# Negative variable index resolves to the correct element, so a wrong
# expected value must be caught (companion to github_4926).
a = [1, 2, 3]
i = 1
assert a[-i] == 999
