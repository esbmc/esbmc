# Issue #8920: list(gen(*xs)) is inlined only behind a single length guard,
# so a generator with defaults among the starred slots is still refused when
# the length of xs is not known before the run.
def gen(a, b=1):
    yield a
    yield b


def t(xs, k):
    assert list(gen(*xs)) == k


t([5], [5, 1])
t([5, 6], [5, 6])
