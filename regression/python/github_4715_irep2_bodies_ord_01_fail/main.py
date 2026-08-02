def f(x: int) -> bool:
    return True if 97 <= ord(x) else False


if __name__ == "__main__":
    assert f(5) == True
