import random


# Program 1
def sum_numbers1(a: int, b: int) -> int:
    return a + b


# Program 2
def sum_numbers2(a: int, b: int) -> int:
    result: int = a
    result += b
    return result


x: int = random.randint(1, 10000)
y: int = random.randint(1, 10000)
assert sum_numbers1(x, y + 1) == sum_numbers2(x, y)
