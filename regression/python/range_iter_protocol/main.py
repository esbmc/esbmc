# Explicit iterator protocol on statically-sized sequences.
# seq.__iter__() returns an iterator; next(it) / it.__next__() consume it.
r = range(4)
i1 = r.__iter__()
i2 = r.__iter__()

# Two independent iterators over the same range keep separate positions.
a1 = next(i1)
a2 = next(i1)
b1 = i2.__next__()
assert a1 == 0
assert a2 == 1
assert b1 == 0

# range with explicit start/step.
rs = range(2, 10, 3)
s = rs.__iter__()
s0 = next(s)
s1 = next(s)
s2 = next(s)
assert s0 == 2
assert s1 == 5
assert s2 == 8

# tuple iterator, mixing next() and .__next__().
xs = (10, 20, 30)
it = xs.__iter__()
c0 = next(it)
c1 = it.__next__()
c2 = next(it)
assert c0 == 10
assert c1 == 20
assert c2 == 30
