def func2():
    return 5


def func1(n):
    num = 0
    while num < n:
        yield func2
        num += 1


n = 3
for i in func1(n):
    try:
        a += i()
    except NameError:
        a = i()
assert a == 5 * n + 1
