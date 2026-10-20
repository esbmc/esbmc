def outer(flag):
    def nested():
        if flag:
            return "s"
        return 3

    return nested()


def through_local(flag):
    def nested():
        if flag:
            return "s"
        return 3

    r = nested()
    return r


def captured(flag, n: int):
    def nested():
        if flag:
            return "s"
        return n

    return nested()


def pick(flag):
    if flag:
        return "s"
    return 3


def forward(flag):
    return pick(flag)


assert outer(False) == 3
assert through_local(True) == "s"
assert captured(False, 5) == 5
assert forward(False) == 3
