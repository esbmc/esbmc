def unsupported(x: int) -> int:
    if x > 0:
        return 1
    match x:
        case int():
            return 0
    return 2


assert unsupported(1) == 1
