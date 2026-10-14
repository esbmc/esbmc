def f(s: str) -> bool:
    return True if 97 <= ord(s) else False


if __name__ == "__main__":
    assert f("a") == True
