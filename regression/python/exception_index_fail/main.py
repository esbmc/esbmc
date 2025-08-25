def check_index(idx:int) -> int:
    if idx < 0 or idx > 10:
        raise IndexError("Index out of range")
    return idx

result = 1
try:
    result = check_index(15)
except ValueError as e:
    assert 1
except IndexError as e:
    assert 0
