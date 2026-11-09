# Issue #8920: list(gen(*xs)) with defaults among the starred slots. Each
# length of xs gets its own copy of t (#9797), so the length is known there.
def gen(a, b=1):
    yield a
    yield b


def t(xs, k):
    assert list(gen(*xs)) == k


t([5], [5, 1])
t([5, 6], [5, 6])
