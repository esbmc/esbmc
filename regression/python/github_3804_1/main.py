# reversed(range(n)) - basic case from the issue
def f():
    a = [0] * 5
    for i in reversed(range(4)):
        x = a[i]

a: int = 0
for i in reversed(range(5)):
    a = i

assert a == 0

# Collect values in reverse
result: int = 0
for i in reversed(range(4)):
    result = result * 10 + i

# reversed(range(4)) gives: 3, 2, 1, 0
# result = ((((0*10+3)*10+2)*10+1)*10+0) = 3210
assert result == 3210
