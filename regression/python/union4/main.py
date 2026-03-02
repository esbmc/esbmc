def returns_union(b: bool) -> int | bool:
    if b:
        return 1
    else:
        return False


assert returns_union(True) == 1
