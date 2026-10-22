# An argument before the star that has an effect is evaluated before the
# length check raises, as in CPython. KNOWNBUG: the frontend emits the check
# first, with the old lowering as with this one. Two lengths keep the length of
# rest unknown before the run.
def add3(a, b, c):
    return a + b + c


def note(seen):
    seen.append(1)
    return 1


def call(rest):
    seen = []
    caught = False
    try:
        add3(note(seen), *rest)
    except TypeError:
        caught = True
    return caught and len(seen) == 1


assert call([2])
assert call([2, 3, 4])
