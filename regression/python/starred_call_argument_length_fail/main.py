# A starred list longer than the remaining parameters raises TypeError
# rather than having its extra element dropped.
def add(a, b):
    return a + b


def call(xs):
    return add(*xs)


assert call([1, 2, 3]) == 3
