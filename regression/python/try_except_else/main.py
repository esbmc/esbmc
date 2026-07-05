# The try `else` clause runs only when the try body completes without an
# exception (previously it was silently dropped). It must be skipped when an
# exception is raised, and its own exceptions must NOT be caught by this try's
# handlers.
x = 0
try:
    pass
except ValueError:
    x = 1
else:
    x = 2
assert x == 2                 # no exception -> else runs

y = 0
try:
    raise ValueError()
except ValueError:
    y = 1
else:
    y = 2
assert y == 1                 # exception -> else skipped

r = []
try:
    r.append(1)
except ValueError:
    r.append(9)
else:
    r.append(2)
assert r == [1, 2]

# Multiple handlers plus else.
z = 0
try:
    pass
except ValueError:
    z = 1
except KeyError:
    z = 2
else:
    z = 3
assert z == 3

# In a loop the guard must reset each iteration: alternating raise/no-raise
# gives one else-run and one catch per pair.
else_runs = 0
caught = 0
for i in range(4):
    try:
        if i % 2 == 0:
            raise ValueError()
    except ValueError:
        caught += 1
    else:
        else_runs += 1
assert else_runs == 2 and caught == 2

# return in the try body skips the else.
def pick(v):
    try:
        if v:
            return "early"
    except ValueError:
        return "err"
    else:
        return "else"
    return "fell"
assert pick(True) == "early"
assert pick(False) == "else"
