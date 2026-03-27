def return_none(condition:bool) -> int:
    a = None
    if condition:
        a = 1
    return a

result = return_none(False)
assert result == 1