# The raise leaves a frame and is caught at the call site.
class MyError(Exception):
    pass


def boom():
    raise MyError()


caught = 0
try:
    boom()
    caught = 99
except:
    caught = 1
assert caught == 1
