def unsupported(x: int) -> int:
    match x:
        case 1:
            return 1
    return 0


class A:
    pass


name = "x"
assert getattr(A(), name, 1) == 1
