def inc(v):
    return v + 1


def other():

    def inc(w):
        return w

    return inc(3)


z = inc("s") if nondet_bool() else inc(2)
y = other()
