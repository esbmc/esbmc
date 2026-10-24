def m(hint) -> bool:
    return hint is type(None)


assert m(int)
