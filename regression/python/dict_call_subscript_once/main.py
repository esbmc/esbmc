calls = 0


def counts():
    global calls
    calls += 1
    r = {"B": 5}
    return r


assert counts()["B"] == 5
assert calls == 1
