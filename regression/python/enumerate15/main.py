# Symbolic start argument for enumerate
start = int(input())  # Symbolic integer
numbers = [10, 20, 30]

for i, x in enumerate(numbers, start):
    assert x >= 10  # trivial assertion to exercise loop


