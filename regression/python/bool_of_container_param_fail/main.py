# bool() of a list parameter relabelled its pointer, so an empty list was
# proved true.
def nonempty(xs: list) -> bool:
    return bool(xs)


assert nonempty([])
