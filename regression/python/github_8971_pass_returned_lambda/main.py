def make(k):
    return lambda x: x + k


flag = nondet_bool()
if flag:
    k, x = 1, 2
else:
    k, x = "a", "b"
assert make(k)(x) == x + k
