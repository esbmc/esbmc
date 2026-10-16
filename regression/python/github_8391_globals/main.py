def unsupported(x: int) -> int:
    match x:
        case int():
            return 1
    return 0


assert len(globals()) > 0
