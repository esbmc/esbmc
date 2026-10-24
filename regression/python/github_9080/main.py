# Issue #9080: an exception raised in a try's else clause skips that try's
# handlers, runs its finally, and reaches the enclosing handler.
log = 0
try:
    try:
        log = log * 10 + 1
    except ValueError:
        log = log * 10 + 2
    else:
        raise ValueError("e")
    finally:
        log = log * 10 + 4
except ValueError:
    log = log * 10 + 5
assert log == 145

log = 0
try:
    try:
        log = log * 10 + 1
    except:
        log = log * 10 + 2
    else:
        raise KeyError("k")
    finally:
        log = log * 10 + 4
except KeyError:
    log = log * 10 + 5
assert log == 145


def f() -> int:
    try:
        y = 1
    except ValueError:
        return 2
    else:
        raise ValueError("e")
    return 0


caught = False
try:
    f()
except ValueError:
    caught = True
assert caught

n = 0
for i in range(3):
    try:
        if i == 1:
            break
    except ValueError:
        n = 100
    else:
        n = n + 1
assert n == 1

f = 0
for i in range(3):
    try:
        if i == 2:
            raise ValueError("v")
    except ValueError:
        f = f + 100
    else:
        if i == 1:
            continue
        f = f + 1
    finally:
        f = f + 10
assert f == 131

d = {"a": 1}
try:
    try:
        pass
    except KeyError:
        f = 0
    else:
        d["b"]
    finally:
        f = f + 1
except KeyError:
    f = f + 1000
assert f == 1132
