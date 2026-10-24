# github #8890: the str path must reach the int comparison.
def outer(flag):
    def nested():
        if flag:
            return "s"
        return 3

    return nested()


c = nondet_bool()
d = outer(c)
assert d == 3
