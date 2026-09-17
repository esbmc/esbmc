# A raise from inside a handler is caught by the enclosing try.
class MyError:
    pass


n = 0
try:
    try:
        raise MyError()
    except:
        n = 1
        raise MyError()
except:
    n = n + 10
assert n == 11
