# reversed(range(start, stop, step)) - 3-arg form with constant step
# reversed(range(2, 10, 3)) = [8, 5, 2]
last: int = -1
for i in reversed(range(2, 10, 3)):
    last = i
assert last == 2

# reversed(range(0, 10, 2)) = [8, 6, 4, 2, 0]
first: int = -1
for i in reversed(range(0, 10, 2)):
    if first == -1:
        first = i
assert first == 8

# reversed(range(10, 2, -3)) = [4, 7, 10]
count: int = 0
for i in reversed(range(10, 2, -3)):
    count = count + 1
assert count == 3

# Empty range with step
empty_count: int = 0
for i in reversed(range(5, 5, 1)):
    empty_count = empty_count + 1
assert empty_count == 0
