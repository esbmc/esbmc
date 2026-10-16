# g()[0] takes its element type from g's list[list[int]] return annotation, so
# == compares it by content rather than by identity.
def g() -> list[list[int]]:
    return [[5], [6]]


assert g()[0] == [5]
