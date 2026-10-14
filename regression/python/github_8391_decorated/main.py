def register(f):
    return f


@register
def unsupported(x: int) -> int:
    match x:
        case 1:
            return 1
    return 0


assert register(1) == 1
