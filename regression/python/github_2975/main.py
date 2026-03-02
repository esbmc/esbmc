def foo(s: str | int) -> int:
    if isinstance(s, int):
        return s

    if isinstance(s, str):
        if 'X' not in s:
            return 42
        else:
            return 17

    return -1


assert foo(4) == 4
assert foo('') == 42
