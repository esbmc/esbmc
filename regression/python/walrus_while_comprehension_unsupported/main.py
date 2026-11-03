# A walrus inside a comprehension in a while-loop condition is lowered with
# the comprehension, not with the condition prelude; ESBMC refuses it.
i = 0
while len({0: (y := i) for _ in range(1)}) and i < 3:
    i += 1
assert y == 3
