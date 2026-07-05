# reversed() over a list, exercised through CPython-valid patterns that ESBMC
# models: slice assignment, for-loop consumption, and list(reversed(...)).

# 1. Slice-assignment RHS (the next_permutation pattern): reverse a tail.
xs = [10, 20, 30, 40, 50]
xs[1:] = reversed(xs[1:])
assert xs == [10, 50, 40, 30, 20]

# 2. Materialise via list(reversed(...)).
ints = [1, 2, 3, 4]
assert list(reversed(ints)) == [4, 3, 2, 1]

# 3. for-loop consumption preserves reverse order.
acc = 0
weight = 1
for e in reversed([1, 2, 3]):
    acc = acc + e * weight
    weight = weight * 10
assert acc == 123  # 3*1 + 2*10 + 1*100

# 4. Float and string element types.
floats = [1.5, 2.5, 3.5]
assert list(reversed(floats)) == [3.5, 2.5, 1.5]
chars = ["a", "b", "c"]
assert list(reversed(chars)) == ["c", "b", "a"]
