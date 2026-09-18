def pick(flag):
    if flag:
        return 1
    return None


a = pick(True)
b = pick(False)
assert a == 1
assert b is None
assert a is not None
