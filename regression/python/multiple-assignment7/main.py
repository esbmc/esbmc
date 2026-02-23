def f():
    print("called")
    return (9, 8)

(x, y) = (u, v) = f()
assert (x, y) == (9, 8)
assert (u, v) == (9, 8)
