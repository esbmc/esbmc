# A walrus inside a lambda in a while-loop condition binds in the lambda's own
# scope each time it is called; ESBMC refuses with a clean diagnostic rather
# than risk an unsound verdict.
i = 0
while (lambda: (k := i))() < 3:
    i += 1
assert i == 3
