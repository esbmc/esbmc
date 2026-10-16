# A list literal bound inside a try, or in a while loop whose break skips its
# else arm, is not what the code after the statement reads.
a = nondet_int()
l = [0]
try:
    if a > 0:
        raise ValueError()
    l = [1]
except ValueError:
    l = [2]
assert (l[0] == 2) == (a > 0)

m = [0]
i = 0
while i < 1:
    if a > 0:
        m = [1]
        break
    i += 1
else:
    m = [2]
assert (m[0] == 2) == (a <= 0)
