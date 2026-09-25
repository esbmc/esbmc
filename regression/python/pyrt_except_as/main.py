# `as e` binds the object that was thrown, not merely a name.
class MyError(Exception):
    pass


got = 0
try:
    raise MyError()
except MyError as e:
    got = isinstance(e, MyError)
assert got
