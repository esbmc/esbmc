# Control flow through a raise: the body runs up to it, the statement after it
# does not, and the handler takes over. What is thrown is not observed here --
# CPython reaches the handler by converting this raise into a TypeError, since
# MyError does not derive from BaseException, where the runtime throws the
# object itself. The observable behaviour is the same.
class MyError:
    pass


log = 0
try:
    log = 1
    raise MyError()
    log = 99
except:
    log = log + 10
assert log == 11

# No exception: the handler is skipped, and a name bound in the body survives.
caught = 0
try:
    x = 1
    caught = x
except:
    caught = 99
assert caught == 1
