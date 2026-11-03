def f() -> int:
    import string
    string = "a b"
    return len(string.split())


assert f() == 2
