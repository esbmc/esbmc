# Companion to github_4909: the mutated value is resolved correctly, so a
# wrong expected value must be caught.
Y = 1
X = Y
a = [X]
for i in range(1):
    a[i] = -1
assert a[0] == 999
