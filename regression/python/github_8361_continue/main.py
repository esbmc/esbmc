# `continue` re-evaluates the condition, assignment included; the name keeps
# the value of the final failing evaluation after the loop.
i = 0
odd = 0
while (j := i) < 5:
    i += 1
    if j % 2 == 0:
        continue
    odd += 1
assert odd == 2
assert j == 5
