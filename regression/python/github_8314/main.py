def add(a, b):
    return a + b


r1 = add(5, 10)
r3 = add("Hello", "World")
if nondet_bool():
    x = r1
else:
    x = r3
half = x / 2
