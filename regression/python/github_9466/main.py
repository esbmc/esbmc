# github issue #9466: a dict comprehension with two for clauses
x = 10
y = 20
b = [1, 2]
e = [3, 4]
m = {x: y for x in b for y in e}
assert m[1] == 4 and m[2] == 4

n = {x * 10 + y: x + y for x in b for y in e if y > x + 1}
assert x == 10 and y == 20
assert len(n) == 3
assert n[13] == 4 and n[14] == 5 and n[24] == 6
assert 23 not in n
