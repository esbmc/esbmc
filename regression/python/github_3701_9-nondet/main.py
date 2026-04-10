flag = nondet_bool()

def gen():
    if flag:
        yield 1
    else:
        yield 2

g = gen()
x = next(g)

# Real Python: x == 1 or x == 2
assert x == 1 or x == 2
