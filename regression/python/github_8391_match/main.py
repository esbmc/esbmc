def unused(x):
    match x:
        case 1:
            return 2
    return 0


def used() -> int:
    return 1


assert used() == 1
