# A union with None keeps its pre-#7876 typing: were it opaque, the 0 argument
# would read as None and f would return 0.
def f(y: int | list[int] | None) -> int:
    if y is None:
        return 0
    return y + 1


assert f(0) == 1
