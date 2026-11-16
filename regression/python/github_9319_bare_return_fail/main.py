def text():
    return "s"


def make(k):
    if k:
        return
    return text


def unused():
    h = make(0)
    return h()


def inc(v):
    return v + 1


x = inc(True)
y = inc("s")
