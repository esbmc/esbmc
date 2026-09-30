def f(x):
    if x % 5 == 0:
        x = 5
    else:
        x = True
    return x

def g(a):
    a.append(3)
    return a[0]

b = lambda y: y + 1
c = (lambda z: z*2)
d = None
e = lambda t: t is None
