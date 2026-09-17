# Members are unique, so a literal naming one twice holds it once, and add()
# on a member already there changes nothing. Equality ignores order.
s = {1, 2, 3}
assert len(s) == 3
assert 2 in s
assert 9 not in s

assert len({1, 1, 2}) == 2

e = set()
assert len(e) == 0
e.add(5)
e.add(5)
assert len(e) == 1
assert 5 in e

# discard() is silent about a member that is not there; remove() is not.
e.discard(5)
assert len(e) == 0
e.discard(99)

f = {1, 2}
f.remove(1)
assert len(f) == 1

assert {1, 2} == {2, 1}
assert {1, 2} != {1, 3}

total = 0
for v in {1, 2, 3}:
    total = total + v
assert total == 6

assert isinstance({1}, set)
