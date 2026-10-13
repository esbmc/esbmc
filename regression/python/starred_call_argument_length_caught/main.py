# A starred list shorter than the remaining parameters raises a TypeError
# that the program can catch.
def add(a, b):
    return a + b


def call(xs):
    caught = False
    try:
        add(*xs)
    except TypeError:
        caught = True
    return caught


assert call([1])
