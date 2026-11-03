def f(v: list[int]) -> int:
    match v:
        case [1, x]:
            return x
    return 0


assert f([1, 2]) == 2
