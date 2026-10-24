# github #8890: a function returning a nested int-or-str call is typed by
# both branches, not by the nested function's last return.
def outer(flag):
    def nested():
        if flag:
            return "s"
        return 3

    return nested()


c = nondet_bool()
d = outer(c)
if c:
    assert d == "s"
else:
    assert d == 3
