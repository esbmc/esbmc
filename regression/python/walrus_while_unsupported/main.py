# A walrus in a while-loop condition would need re-binding each iteration;
# ESBMC refuses with a clean diagnostic rather than risk an unsound verdict.
i = 0
while (k := i) < 3:
    i += 1
assert i == 3
