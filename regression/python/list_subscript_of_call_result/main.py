# x takes its type from g's list[list[int]] return annotation, so == compares
# x by content rather than by identity.
def g() -> list[list[int]]:
    return [[5], [6]]


x = g()[0]
assert x == [5]
