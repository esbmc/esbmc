a = range
b = a

count: int = 0
for i in b(3):
    count = count + 1

assert count == 3
