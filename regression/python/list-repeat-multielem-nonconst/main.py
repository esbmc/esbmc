def f(n: int) -> int:
    a = [1, 2] * n
    return len(a)


if __name__ == "__main__":
    assert f(3) == 6
