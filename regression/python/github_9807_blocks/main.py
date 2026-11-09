def g():
    return 1


while True:
    try:
        raise ValueError("x")
    except ValueError:
        f = g
    finally:
        h = g
    break
r = f()
s = h()
assert r == 1 and s == 1
