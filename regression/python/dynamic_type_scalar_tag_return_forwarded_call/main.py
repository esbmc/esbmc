def nested(flag):
    if flag:
        return "s"
    return 3


def outer(flag):
    return nested(flag)


b = outer(False)
assert b == 3
