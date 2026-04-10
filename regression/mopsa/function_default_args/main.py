def f(x, y):
    return x + y

def g(x, y, z = 10):
    return x + y + z

res = f(1, 2) + \
      g(3, 4) + \
      g(5, 6, 7)
