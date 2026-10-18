# No handler matches, so the exception keeps propagating and escapes.
class MyError(Exception):
    pass


class Other(Exception):
    pass


try:
    raise MyError()
except Other:
    pass
