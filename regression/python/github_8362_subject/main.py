calls = 0


def next_value() -> int:
    global calls
    calls += 1
    return calls


match next_value():
    case 2:
        assert False
    case 1:
        flag = None
    case _:
        assert False

match flag:
    case True:
        assert False
    case None:
        done = 1

assert calls == 1
assert done == 1
