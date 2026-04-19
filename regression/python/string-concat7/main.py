def f():
    s = ""
    while True:
        s = "A" + s
        break
    assert s == "A"

f()
