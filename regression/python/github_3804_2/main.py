# reversed(range(start, stop)) - 2-arg form
last: int = -1
for i in reversed(range(2, 7)):
    last = i

# reversed(range(2,7)) gives: 6, 5, 4, 3, 2 — last is 2
assert last == 2

# Verify first seen value is stop-1
first: int = -1
for i in reversed(range(3, 8)):
    if first == -1:
        first = i
assert first == 7

# Empty range reversed is empty
count: int = 0
for i in reversed(range(5, 2)):
    count = count + 1
assert count == 0
