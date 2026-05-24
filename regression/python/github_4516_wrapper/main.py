def my_range(n: int):
    return range(n)


count: int = 0
for i in my_range(5):
    count = count + 1

assert count == 5
