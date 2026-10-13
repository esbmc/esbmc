def unsupported(x: int) -> int:
    if x > 0:
        return 1
    match x:
        case 0:
            return 0
    return 2


assert unsupported(1) == 1
