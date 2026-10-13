def unsupported(x: int) -> int:
    match x:
        case 1:
            return 1
    return 0


assert len(globals()) > 0
