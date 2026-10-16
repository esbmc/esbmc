def unsupported(x: int) -> int:
    match x:
        case int():
            return 1
    return 0


f = unsupported
assert f(1) == 2
