def inc(v):
    return v + 1


class C:

    def inc(self, v):
        return v


z = inc("s") if nondet_bool() else inc(2)
