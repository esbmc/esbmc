# x is reassigned int/str inside a loop body across an if/else.

x = 0
i = 0
while i < 3:
    if nondet_bool():
        x = 1
    else:
        x = "a"
    i += 1
assert x == 1 or x == "a"
