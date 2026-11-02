x: list[int] = nondet_list(3, nondet_bool())
y = [1]
r = x and y
assert r == [1]
