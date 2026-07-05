# tuple(iterable) over lists and tuples. The tuple is modelled as the
# underlying sequence object, so equality, len(), subscript, and iteration
# route through the list machinery (#4807, humaneval_33 pattern).

def ident(l: list):
    l = list(l)
    return l

# tuple of a list literal
assert tuple([1, 2, 3]) == tuple([1, 2, 3])

# tuple of a list variable
a = [1, 2, 3]
assert tuple(a) == tuple(a)

# tuple of a list-returning call (the humaneval_33 assertion shape)
assert tuple(ident([1, 2, 3])) == tuple(ident([1, 2, 3]))
assert tuple(ident([4, 5, 6]))[0] == 4

# tuple of a tuple is the tuple itself (CPython identity)
t = tuple((1, 2))
assert t == (1, 2)

# tuple of a tuple-returning call
def pair():
    return (1, 2)

assert tuple(pair()) == (1, 2)

# tuple() copies: mutating the source list afterwards must not show
# through the tuple (CPython snapshot semantics)
src = [1, 2]
snap = tuple(src)
src[0] = 9
assert snap[0] == 1
assert src[0] == 9

# len, subscript, and iteration over a tuple built from a list
u = tuple([4, 5, 6])
assert len(u) == 3
assert u[0] == 4
assert u[2] == 6
total = 0
for x in u:
    total = total + x
assert total == 15

print("ok")
