def conditional_int(flag: bool) -> "int":
    if flag:
        return 1
    else:
        return 0


def conditional_str(value: int) -> "str":
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return "zero"


def conditional_bool(x: int) -> "bool":
    if x % 2 == 0:
        return 1
    return 0


result1 = conditional_int(True)
result2 = conditional_str(-5)
result3 = conditional_bool(4)
