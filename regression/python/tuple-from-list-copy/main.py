# Split from tuple-from-list: tuple-of-tuple identity and the snapshot
# semantics of tuple() over a list.

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

print("ok")
