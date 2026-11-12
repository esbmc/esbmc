def raises_type_error(a: str, b: str, op: str) -> bool:
    try:
        if op == "-":
            a - b
        elif op == "/":
            a / b
        elif op == "//":
            a // b
        else:
            a**b
    except TypeError:
        return True
    return False


def sub_int(a: str, n: int) -> bool:
    try:
        a - n
    except TypeError:
        return True
    return False


def concat(a: str, b: str) -> str:
    return a + b


def less(a: str, b: str) -> bool:
    return a < b


assert raises_type_error("x", "y", "-")
assert raises_type_error("x", "y", "/")
assert raises_type_error("x", "y", "//")
assert raises_type_error("x", "y", "**")
assert sub_int("x", 1)
assert concat("x", "y") == "xy"
assert less("a", "b")
