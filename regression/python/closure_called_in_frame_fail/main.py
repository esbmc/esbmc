def outer(flag):
    def nested():
        if flag:
            return 1
        return 3

    return nested()


c = outer(False)
assert c == 3
b = outer(True)
assert b == 3
