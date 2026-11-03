def inc(v):
    return v + 1


z = inc("s") if nondet_bool() else inc(2)
