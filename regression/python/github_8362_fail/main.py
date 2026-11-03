def f(v: str) -> int:
    match v:
        case "a":
            return 1
        case "b" | "c":
            return 3
        case _:
            return 2


assert f("c") == 2
