def outer(flag):
    def nested():
        if flag:
            return "s"
        return 3

    return nested()


b = outer(True)
assert b == 3
