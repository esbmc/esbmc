def test_binary(x: str) -> bool:
    return int(x, 2) == 5


assert test_binary("110")  # 6 in binary, not 5
