import random

counter = 0

def sum_with_side_effect(numbers: list[int]) -> int:
    global counter
    total = 0
    for num in numbers:
        total += num
    counter += 1  # side effect
    return total

def sum_no_side_effect(numbers: list[int]) -> int:
    total = 0
    for num in numbers:
        total += num
    return total

x = [random.randint(1, 100) for _ in range(random.randint(1, 10))]

assert sum_with_side_effect(x) == sum_no_side_effect(x)
