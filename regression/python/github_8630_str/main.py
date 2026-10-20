def f(a):
    return a * 2


xs = [1, 2, 3]
c = [f(a) for a in [str(b) for b in xs]]
assert c[0] == "11"
